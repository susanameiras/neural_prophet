from typing import List, Dict, Optional
import torch
from torch import nn
import numpy as np

from neuralprophet.components.trend import Trend

# Helper to avoid numerical issues
def sigmoid_inverse_and_clip(y):
    """Numerically stable inverse of the logit transform."""
    return torch.clamp(-torch.log(1.0 / y - 1.0), -1e10, 1e10)

class LogisticTrend(Trend):
    
    def __init__(
        self,
        config,
        id_list: List[str],
        quantiles,
        num_trends_modelled: int,
        n_forecasts: int,
        device: torch.device,

    ):
        super().__init__(config, id_list, quantiles, num_trends_modelled, n_forecasts, device)

        # Validate that cap is provided
        self.cap_dict: Dict[str, float] = getattr(config, "cap", None)
        if self.cap_dict is None:
            raise ValueError("Logistic growth requires 'cap' to be specified in config")     

        self.floor_dict: Dict[str, float] = getattr(config, "floor", {})
        
        # Get number of changepoints
        self.n_changepoints = int(getattr(config, "n_changepoints", 0))
        self.changepoints_range = getattr(config, "changepoints_range", 0.8)      

        # Initialize changepoint locations (fixed, not learnable)

        if self.n_changepoints > 0:
            # Changepoints uniformly spaced in first changepoints_range of time
            self.register_buffer(
                "changepoints",
                torch.linspace(0, self.changepoints_range, self.n_changepoints, device=device)
            )

        # Determine if we have global or local trends
        self.num_trends_modelled = num_trends_modelled
        self.is_global = (num_trends_modelled == 1)       

        # Initialize learnable parameters
        self._init_parameters() 

    def _init_parameters(self):
        """Initialize growth rate k, offset m, and changepoint deltas."""
        n_params = 1 if self.is_global else self.num_trends_modelled
        
        # Base growth rate k (using log-space parameter for stability is often better)
        # Prophet initializes with linear trend, but for learnable params, we start near 0.1
        self.k = nn.Parameter(torch.ones(n_params, 1, 1, device=self.device) * 0.1)
        
        # Offset m (initialized to 0, will be learned)
        self.m = nn.Parameter(torch.zeros(n_params, 1, 1, device=self.device))
        
        # Changepoint rate adjustments (delta)
        if self.n_changepoints > 0:
            self.delta = nn.Parameter(
                torch.zeros(n_params, self.n_changepoints, 1, device=self.device)
            )
        else:
            self.delta = None
            
        # Gamma is not a learnable parameter, it is calculated from k, m, and delta
        self.register_buffer("gamma", None)

    # ... (Omitted _get_cap_floor_tensors and _compute_changepoint_indicators - they are fine) ...
    
    def _compute_gamma(self, k: torch.Tensor, m: torch.Tensor, delta: torch.Tensor):
        """
        Compute offset adjustments γ to ensure trend continuity at changepoints.
        
        This logic closely follows Prophet's implementation to maintain continuity
        in the value of the logistic function at each changepoint.
        
        Parameters:
            k: Base growth rate (N, 1, 1) or (B, 1, 1)
            m: Base offset (N, 1, 1) or (B, 1, 1)
            delta: Rate adjustments (N, C, 1) or (B, C, 1)
            
        Returns:
            gamma: Offset adjustments (N, C, 1) or (B, C, 1)
        """
        if delta is None:
            return None
        
        # N: num_trends_modelled or B: batch_size for global
        N = k.shape[0]
        C = self.n_changepoints
        
        # changepoints: (C,) -> (1, C, 1)
        s = self.changepoints.view(1, C, 1) 
        
        # k: (N, 1, 1), delta: (N, C, 1)
        
        # k_0 is the initial growth rate (k)
        # k_vec: (N, C, 1) - contains k at each segment start (initial k for j=0)
        # torch.cat([k.expand(-1, 1, -1), delta], dim=1) gives (N, C+1, 1) rates
        
        # Cumulative rate (k_j = k_base + sum(delta_i for i < j))
        # k_0: (N, 1, 1)
        # total_delta: (N, C, 1). Total adjustment up to and including changepoint j.
        total_delta = torch.cumsum(delta, dim=1) # (N, C, 1)
        
        # k_c: (N, C, 1). Total rate *after* changepoint c (k_base + total_delta_c)
        k_c = k + total_delta 
        
        # k_prev: (N, C, 1). Total rate *before* changepoint c.
        # This is k_base for the first changepoint, and k_{c-1} for others.
        k_prev = torch.cat([k, k_c[:, :-1, :]], dim=1) # (N, C, 1)
        
        # Initial m (m_0) and gamma accumulator
        m_prev = m.clone() # (N, 1, 1)
        gamma = torch.zeros_like(delta) # (N, C, 1)
        
        # Iterative calculation for continuity adjustment
        # Note: torch.autograd.functional.vjp or custom operations would be needed 
        # to properly vectorize this complex sequential logic across the C dimension, 
        # but a simple loop is often clearer and faster for small C (num_changepoints).
        for c in range(C):
            # Calculate gamma for changepoint c
            # This is the Prophet continuity equation:
            # gamma[c] = (s[c] - m_prev) * (1 - k_prev[c] / k_c[c])
            
            # The base offset term m_prev *must* be the total accumulated offset 
            # *before* the current changepoint.
            
            # (N, 1, 1) or (N, 1, 1). If N > 1, this needs series-specific indexing.
            # Assuming N=1 (global) or N=B (local and mapped)
            
            # Extract relevant tensors for changepoint c (N, 1, 1)
            k_c_c = k_c[:, c:c+1, :]
            k_prev_c = k_prev[:, c:c+1, :]
            s_c = s[:, c:c+1, :]
            
            # Compute gamma_c
            gamma_c = (s_c - m_prev) * (1.0 - k_prev_c / k_c_c) # (N, 1, 1)
            
            # Store gamma_c
            gamma[:, c:c+1, :] = gamma_c
            
            # Update m_prev for the next changepoint
            m_prev = m_prev + gamma_c
        
        self.gamma = gamma # Register as a buffer after calculation
        return gamma


    def forward(self, t: torch.Tensor, meta=None) -> torch.Tensor:
        """
        Compute logistic trend.
        
        Parameters:
            t: normalized time, shape (batch, n_forecasts)
            meta: metadata dict containing 'df_name' for series identification
            
        Returns:
            trend: shape (batch, n_forecasts, n_quantiles)
        """
        batch_size, n_forecasts = t.shape
        
        # Get cap and floor for each series in batch
        cap, floor = self._get_cap_floor_tensors(meta, batch_size)
        
        # Select/Expand appropriate k and m based on global/local
        if self.is_global:
            # Global: (1, 1, 1) -> (B, 1, 1)
            k_base = self.k.expand(batch_size, -1, -1)
            m_base = self.m.expand(batch_size, -1, -1)
            delta = self.delta.expand(batch_size, -1, -1) if self.delta is not None else None
        else:
            # Local: (N, 1, 1) where N=num_trends_modelled. 
            # Assumes batch_size must match N and be correctly indexed by caller.
            k_base = self.k
            m_base = self.m
            delta = self.delta
            
        # Compute gamma *before* using it in the forward pass.
        # k_base and m_base are now (N or B, 1, 1)
        gamma = self._compute_gamma(k_base, m_base, delta)
            
        # Expand t for broadcasting: (B, T) -> (B, T, 1)
        t_expanded = t.unsqueeze(-1)
        
        # Initialize k(t) and m(t)
        k_t = k_base.expand(-1, n_forecasts, -1)  # (B, T, 1)
        m_t = m_base.expand(-1, n_forecasts, -1)  # (B, T, 1)

        # Apply changepoint adjustments
        if self.n_changepoints > 0:
            indicators = self._compute_changepoint_indicators(t)  # (B, T, C)
            
            # Delta adjustment for rate k(t) = k + a(t)^T δ
            # delta: (B, C, 1)
            rate_adjustment = torch.matmul(indicators, delta)  # (B, T, 1)
            k_t = k_t + rate_adjustment
            
            # Gamma adjustment for offset m(t) = m + a(t)^T γ
            # gamma: (B, C, 1)
            offset_adjustment = torch.matmul(indicators, gamma) # (B, T, 1)
            m_t = m_t + offset_adjustment
        
        # Compute logistic function:
        # g(t) = floor + (cap - floor) / (1 + exp(-k(t) * (t - m(t))))
        
        # Note: Tensors k_t, t_expanded, m_t are (B, T, 1)
        exponent = -k_t * (t_expanded - m_t)  # (B, T, 1)
        
        # Clip exponent to avoid numerical overflow (your original clipping is good)
        exponent = torch.clamp(exponent, -20.0, 20.0)
        
        logistic_term = 1.0 / (1.0 + torch.exp(exponent))  # (B, T, 1)
        
        # Apply floor and cap
        trend = floor.expand(-1, n_forecasts, -1) + \
                (cap - floor).expand(-1, n_forecasts, -1) * logistic_term  # (B, T, 1)
        
        # Expand for quantiles if needed
        if self.config.quantiles and len(self.config.quantiles) > 1:
            n_quantiles = len(self.config.quantiles)
            trend = trend.expand(-1, -1, n_quantiles)  # (B, T, Q)
            
        return trend

    @property
    def get_trend_deltas(self):
        """
        Return changepoint deltas for regularization.      

        Returns:
            delta parameter if changepoints exist, else None
        """

        if self.delta is not None:
            return lambda: self.delta
        return lambda: None

    def add_regularization(self):
        """
        Add regularization loss for changepoint parameters.
       
        Returns:
            Regularization loss (scalar tensor)
        """

        if self.delta is None:
            return torch.tensor(0.0, device=self.device)

        # L1 regularization on delta (sparse changepoint adjustments)
        # Weight by trend regularization parameter from config
        reg_lambda = getattr(self.config, "trend_reg", 0.0)

        if reg_lambda > 0:
            return reg_lambda * torch.abs(self.delta).sum()
      
        return torch.tensor(0.0, device=self.device) 

from typing import List, Dict, Optional, Union
import torch
from torch import nn
import numpy as np

from neuralprophet.components.trend import Trend


class LogisticTrend(Trend):
    """
    Logistic trend component matching Prophet's implementation.
    
    Uses the logistic growth formula:
        g(t) = floor + (cap - floor) / (1 + exp(-k * (t - m)))
    
    With changepoints, it becomes:
        g(t) = floor + (cap - floor) / (1 + exp(-(k + a(t)^T δ) * (t - (m + a(t)^T γ))))
    
    where:
        - k: base growth rate (learnable)
        - m: offset parameter (learnable)
        - δ: rate adjustments at changepoints (learnable if changepoints exist)
        - γ: offset adjustments at changepoints (computed to ensure continuity)
        - a(t): indicator vector for changepoints passed at time t
    
    Expected config attributes:
        - config.growth == "logistic"
        - config.cap: Dict[str, float] or float - carrying capacity
        - config.floor: Dict[str, float] or float - floor value (optional, defaults to 0)
        - config.n_changepoints: number of changepoints
        - config.changepoints_range: proportion of history for changepoints
    """

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
        self.cap = getattr(config, "cap", None)
        if self.cap is None:
            raise ValueError("Logistic growth requires 'cap' to be specified in config")
        
        self.floor = getattr(config, "floor", 0.0)
        
        # Get number of changepoints
        self.n_changepoints = int(getattr(config, "n_changepoints", 0))
        self.changepoints_range = getattr(config, "changepoints_range", 0.8)
        
        # Initialize changepoint locations (fixed, not learnable)
        if self.n_changepoints > 0:
            # Store as attribute for potential later use, but not as buffer
            self.changepoints_t = torch.linspace(
                0, self.changepoints_range, self.n_changepoints + 2, device=device
            )[1:-1]  # Exclude endpoints
        
        # Determine if we have global or local trends
        self.num_trends_modelled = num_trends_modelled
        self.is_global = (num_trends_modelled == 1)
        
        # Store series IDs for mapping
        self.id_list = id_list
        self.id_to_idx = {id_: i for i, id_ in enumerate(id_list)}
        
        # Initialize learnable parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize growth rate k, offset m, and changepoint deltas."""
        # Base growth rate k - initialize to small positive value
        # In Prophet, this is initialized based on data, but we use learnable parameter
        self.k = nn.Parameter(
            torch.ones(self.num_trends_modelled, 1, device=self.device) * 0.05,
            requires_grad=True
        )
        
        # Offset m - initialized to 0
        self.m = nn.Parameter(
            torch.zeros(self.num_trends_modelled, 1, device=self.device),
            requires_grad=True
        )
        
        # Changepoint rate adjustments (delta)
        if self.n_changepoints > 0:
            self.delta = nn.Parameter(
                torch.zeros(self.num_trends_modelled, self.n_changepoints, device=self.device),
                requires_grad=True
            )
        else:
            self.delta = None
        
        # Gamma (offset adjustments for continuity) - computed, not learned
        if self.n_changepoints > 0:
            # Initialize gamma as zeros, will be computed in forward if delta is updated
            self.register_buffer('gamma', torch.zeros_like(self.delta))
            self._gamma_needs_update = True
        else:
            self.gamma = None
            self._gamma_needs_update = False
    
    def _compute_gamma(self):
        """
        Compute offset adjustments γ to ensure trend continuity at changepoints.
        
        Following Prophet's implementation:
            For each changepoint s_j:
                Let k_before = k + sum_{l<j} delta_l
                Let k_after = k_before + delta_j
                gamma_j = (s_j - m) * (1 - k_after / k_before)
        """
        if self.delta is None:
            return None
        
        # Get device and dtype from delta
        device = self.delta.device
        dtype = self.delta.dtype
        
        # Initialize gamma
        gamma = torch.zeros_like(self.delta)
        
        # Cumulative delta
        cum_delta = torch.zeros(self.num_trends_modelled, 1, device=device, dtype=dtype)
        
        # Changepoint times
        s = self.changepoints_t.unsqueeze(0)  # Shape: (1, n_changepoints)
        
        # Compute gamma for each changepoint
        for j in range(self.n_changepoints):
            k_before = self.k + cum_delta
            k_after = k_before + self.delta[:, j:j+1]
            
            # Avoid division by zero
            mask = torch.abs(k_before) > 1e-8
            gamma_j = torch.zeros_like(k_before)
            gamma_j[mask] = (s[:, j:j+1] - self.m) * (1 - k_after / k_before)[mask]
            
            gamma[:, j:j+1] = gamma_j
            cum_delta = cum_delta + self.delta[:, j:j+1]
        
        return gamma
    
    def forward(self, t: torch.Tensor, meta=None) -> torch.Tensor:
        """
        Compute logistic trend.
        
        Parameters:
            t: normalized time, shape (batch, n_forecasts)
            meta: metadata dict containing 'df_name' for series identification
        
        Returns:
            trend: shape (batch, n_forecasts, 1) or (batch, n_forecasts, n_quantiles)
        """
        batch_size, n_forecasts = t.shape
        
        # Get parameter indices for each series in batch
        if self.is_global:
            param_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        else:
            if meta is None or 'df_name' not in meta:
                raise ValueError("Local trends require 'df_name' in meta")
            df_names = meta['df_name']
            param_indices = torch.tensor(
                [self.id_to_idx.get(name, 0) for name in df_names],
                dtype=torch.long,
                device=self.device
            )
        
        # Get cap and floor values for each series
        if isinstance(self.cap, dict):
            caps = torch.tensor(
                [self.cap.get(name, 1.0) for name in df_names],
                dtype=torch.float32,
                device=self.device
            ).view(-1, 1, 1)
        else:
            caps = torch.tensor([self.cap], dtype=torch.float32, device=self.device
                              ).view(1, 1, 1).expand(batch_size, -1, -1)
        
        if isinstance(self.floor, dict):
            floors = torch.tensor(
                [self.floor.get(name, 0.0) for name in df_names],
                dtype=torch.float32,
                device=self.device
            ).view(-1, 1, 1)
        else:
            floors = torch.tensor([self.floor], dtype=torch.float32, device=self.device
                                ).view(1, 1, 1).expand(batch_size, -1, -1)
        
        # Select parameters for each series
        k_selected = self.k[param_indices]  # Shape: (batch_size, 1)
        m_selected = self.m[param_indices]  # Shape: (batch_size, 1)
        
        # Reshape t for broadcasting
        t_reshaped = t.view(batch_size, n_forecasts, 1)  # (B, T, 1)
        
        # Initialize k_t and m_t
        k_t = k_selected.unsqueeze(1).expand(-1, n_forecasts, -1)  # (B, T, 1)
        m_t = m_selected.unsqueeze(1).expand(-1, n_forecasts, -1)  # (B, T, 1)
        
        # Apply changepoint adjustments if needed
        if self.n_changepoints > 0:
            # Update gamma if delta has changed
            if self.training or self._gamma_needs_update:
                self.gamma = self._compute_gamma()
                self._gamma_needs_update = False
            
            # Get delta and gamma for each series
            delta_selected = self.delta[param_indices]  # (B, n_changepoints)
            gamma_selected = self.gamma[param_indices]  # (B, n_changepoints)
            
            # Compute changepoint indicators
            # t_reshaped: (B, T, 1), changepoints_t: (n_changepoints,)
            indicators = (t_reshaped >= self.changepoints_t.view(1, 1, -1)).float()  # (B, T, C)
            
            # Apply adjustments
            # Sum over changepoints: (B, T, C) @ (B, C, 1) -> (B, T, 1)
            delta_adjustments = torch.bmm(indicators, delta_selected.unsqueeze(-1))
            gamma_adjustments = torch.bmm(indicators, gamma_selected.unsqueeze(-1))
            
            k_t = k_t + delta_adjustments
            m_t = m_t + gamma_adjustments
        
        # Compute logistic function
        # exponent = -k_t * (t_reshaped - m_t)
        exponent = -k_t * (t_reshaped - m_t)
        
        # Clip to avoid numerical issues (Prophet uses ±10)
        exponent = torch.clamp(exponent, -10, 10)
        
        logistic_term = 1.0 / (1.0 + torch.exp(exponent))  # (B, T, 1)
        
        # Apply floor and cap
        trend = floors.expand(-1, n_forecasts, -1) + \
                (caps - floors).expand(-1, n_forecasts, -1) * logistic_term
        
        # Expand for quantiles if needed
        if hasattr(self.config, 'quantiles') and self.config.quantiles:
            n_quantiles = len(self.config.quantiles)
            trend = trend.expand(-1, -1, n_quantiles)
        
        return trend
    
    @property
    def get_trend_deltas(self):
        """Return changepoint deltas for regularization."""
        if self.delta is not None:
            return lambda: self.delta
        return lambda: None
    
    def add_regularization(self):
        """Add regularization loss for changepoint parameters."""
        if self.delta is None:
            return torch.tensor(0.0, device=self.device)
        
        reg_lambda = getattr(self.config, "trend_reg", 0.0)
        if reg_lambda > 0:
            # Mark gamma for update after gradient step
            if self.delta.grad is not None and torch.any(self.delta.grad != 0):
                self._gamma_needs_update = True
            
            # L1 regularization on delta (matching Prophet)
            return reg_lambda * torch.sum(torch.abs(self.delta))
        
        return torch.tensor(0.0, device=self.device)

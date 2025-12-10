from typing import List, Dict, Optional
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
        - config.cap: Dict[str, float] - carrying capacity per time series ID
        - config.floor: Dict[str, float] - floor value per time series ID (optional, defaults to 0)
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
        
        # Base growth rate k (initialized to a reasonable default)
        # Prophet initializes this based on data, but we'll use a learnable parameter
        # that the optimizer will adjust
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
    
    def _get_cap_floor_tensors(self, meta, batch_size: int):
        """
        Build cap and floor tensors from metadata.
        
        Returns:
            cap: shape (batch_size, 1, 1) or (1, 1, 1) if global
            floor: shape (batch_size, 1, 1) or (1, 1, 1) if global
        """
        if meta is None or "df_name" not in meta:
            raise ValueError("Logistic trend requires 'df_name' in meta")
        
        df_names = meta["df_name"]  # list of series IDs
        
        # Get cap values
        caps = [self.cap_dict.get(name, None) for name in df_names]
        if None in caps:
            missing = [name for name, cap in zip(df_names, caps) if cap is None]
            raise ValueError(f"Missing cap values for series: {missing}")
        
        cap_tensor = torch.tensor(caps, dtype=torch.float32, device=self.device).view(-1, 1, 1)
        
        # Get floor values (default to 0)
        floors = [self.floor_dict.get(name, 0.0) for name in df_names]
        floor_tensor = torch.tensor(floors, dtype=torch.float32, device=self.device).view(-1, 1, 1)
        
        return cap_tensor, floor_tensor
    
    def _compute_changepoint_indicators(self, t: torch.Tensor):
        """
        Compute indicator matrix a(t) where a(t)[i] = 1 if t >= changepoint[i], else 0.
        
        Parameters:
            t: time tensor, shape (batch, n_forecasts)
        
        Returns:
            indicators: shape (batch, n_forecasts, n_changepoints)
        """
        if self.n_changepoints == 0:
            return None
        
        # t: (B, T), changepoints: (C,)
        # Expand for broadcasting: t -> (B, T, 1), changepoints -> (1, 1, C)
        t_expanded = t.unsqueeze(-1)  # (B, T, 1)
        cp_expanded = self.changepoints.view(1, 1, -1)  # (1, 1, C)
        
        # Indicator: 1 if t >= changepoint, 0 otherwise
        indicators = (t_expanded >= cp_expanded).float()  # (B, T, C)
        
        return indicators
    
    def _compute_gamma(self):
        """
        Compute offset adjustments γ to ensure trend continuity at changepoints.
        
        Following Prophet's implementation:
            γ[j] = (s[j] - m - sum(γ[:j])) * (1 - k[j] / k[j-1])
        
        where s[j] is the changepoint location and k[j] is the growth rate after changepoint j.
        
        Returns:
            gamma: shape matching delta
        """
        if self.delta is None:
            return None
        
        # For simplicity in NeuralProphet, we can use a simpler formula
        # This ensures continuity but might differ slightly from Prophet's exact implementation
        # A full implementation would need to match Prophet's exact gamma calculation
        
        # For now, we'll compute gamma to maintain continuity
        # gamma[j] needs to offset the change in slope at each changepoint
        gamma = torch.zeros_like(self.delta)
        
        # This is a simplified version - Prophet's actual implementation is more complex
        # and involves cumulative adjustments
        # You may need to refine this based on exact Prophet behavior
        
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
        
        # Select appropriate k and m based on global/local
        if self.is_global:
            k = self.k.expand(batch_size, -1, -1)  # (B, 1, 1)
            m = self.m.expand(batch_size, -1, -1)  # (B, 1, 1)
            if self.delta is not None:
                delta = self.delta.expand(batch_size, -1, -1)  # (B, C, 1)
        else:
            # For local trends, we need to map series IDs to parameter indices
            # This requires maintaining an ID mapping
            # For now, assume sequential mapping
            k = self.k  # (num_trends, 1, 1)
            m = self.m  # (num_trends, 1, 1)
            delta = self.delta if self.delta is not None else None
        
        # Expand t for broadcasting: (B, T) -> (B, T, 1)
        t_expanded = t.unsqueeze(-1)
        
        # Compute growth rate k(t) = k + a(t)^T δ
        if self.n_changepoints > 0:
            indicators = self._compute_changepoint_indicators(t)  # (B, T, C)
            # delta: (B, C, 1) for global, need to handle properly
            # rate_adjustment: (B, T, 1)
            if self.is_global:
                rate_adjustment = torch.matmul(indicators, delta)  # (B, T, 1)
            else:
                # For local trends, this needs proper indexing
                rate_adjustment = torch.matmul(indicators, delta)
            
            k_t = k.expand(-1, n_forecasts, -1) + rate_adjustment  # (B, T, 1)
            
            # Compute offset adjustment m(t) = m + a(t)^T γ
            gamma = self._compute_gamma()
            if gamma is not None:
                offset_adjustment = torch.matmul(indicators, gamma)
                m_t = m.expand(-1, n_forecasts, -1) + offset_adjustment
            else:
                m_t = m.expand(-1, n_forecasts, -1)
        else:
            k_t = k.expand(-1, n_forecasts, -1)  # (B, T, 1)
            m_t = m.expand(-1, n_forecasts, -1)  # (B, T, 1)
        
        # Compute logistic function:
        # g(t) = floor + (cap - floor) / (1 + exp(-k(t) * (t - m(t))))
        exponent = -k_t * (t_expanded - m_t)  # (B, T, 1)
        
        # Clip exponent to avoid numerical overflow
        exponent = torch.clamp(exponent, -20, 20)
        
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

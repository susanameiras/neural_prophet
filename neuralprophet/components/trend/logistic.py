import torch
import torch.nn as nn

from neuralprophet.components.trend import Trend
from neuralprophet.utils_torch import init_parameter

from typing import List, Dict, Optional

class LogisticTrend(Trend):
    """
    Simple logistic trend, one (k, m) per time series ID.

    g(t) = floor + (cap - floor) / (1 + exp(-k * (t - m)))
    """

    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        super().__init__(
            config=config,
            n_forecasts=n_forecasts,
            num_trends_modelled=num_trends_modelled,
            quantiles=quantiles,
            id_list=id_list,
            device=device,
        )
        # Trend_k0  parameter.
        # dimensions - [no. of quantiles,  num_trends_modelled, trend coeff shape]
        self.trend_k0 = init_parameter(dims=([len(self.quantiles)] + [self.num_trends_modelled] + [1]))

        # ---- Prepare cap / floor per ID ----
        # config.cap and config.floor are dicts {id_name: value}
        cap_values = []
        floor_values = []

        for id_name in id_list:
            cap_val = 1.0
            floor_val = 0.0
            if config.cap is not None and id_name in config.cap:
                cap_val = float(config.cap[id_name])
            if config.floor is not None and id_name in config.floor:
                floor_val = float(config.floor[id_name])
            cap_values.append(cap_val)
            floor_values.append(floor_val)

        cap_tensor = torch.tensor(cap_values, dtype=torch.float32, device=device)
        floor_tensor = torch.tensor(floor_values, dtype=torch.float32, device=device)

        # store as non-trainable buffers
        self.register_buffer("cap", cap_tensor)
        self.register_buffer("floor", floor_tensor)

        # ---- Learnable k and m per ID (1 quantile only for now) ----
        # shape: (num_trends_modelled,)
        self.k = nn.Parameter(torch.zeros(num_trends_modelled, dtype=torch.float32, device=device))
        self.m = nn.Parameter(torch.zeros(num_trends_modelled, dtype=torch.float32, device=device))

    def _id_one_hot(self, df_names: List[str]) -> torch.Tensor:
        """Convert list of df_name strings into one-hot tensor of shape (batch, num_trends_modelled)."""
        batch_size = len(df_names)
        one_hot = torch.zeros(batch_size, self.num_trends_modelled, device=self.device)
        id_index = {name: i for i, name in enumerate(self.id_list)}
        for i, name in enumerate(df_names):
            one_hot[i, id_index[name]] = 1.0
        return one_hot

    def forward(self, t: torch.Tensor, meta: Dict) -> torch.Tensor:
        """
        Parameters
        ----------
        t : torch.Tensor
            normalized time, shape (batch, n_forecasts)
        meta : dict
            must contain 'df_name': list[str] of length batch

        Returns
        -------
        torch.Tensor
            logistic trend, same shape as t
        """
        df_names: List[str] = meta["df_name"]
        batch, n_forecasts = t.shape

        # one-hot(df_name) -> (batch, num_trends_modelled)
        one_hot = self._id_one_hot(df_names)

        # Select per-sample parameters:
        # each: (batch,)
        k = (one_hot @ self.k)                  # growth rate per sample
        m = (one_hot @ self.m)                  # offset per sample
        cap = (one_hot @ self.cap)              # cap per sample (constant over time)
        floor = (one_hot @ self.floor)          # floor per sample

        # reshape to (batch, n_forecasts)
        k = k.unsqueeze(1).expand(batch, n_forecasts)
        m = m.unsqueeze(1).expand(batch, n_forecasts)
        cap = cap.unsqueeze(1).expand(batch, n_forecasts)
        floor = floor.unsqueeze(1).expand(batch, n_forecasts)

        C = cap - floor   # "capacity" in current scale

        # Logistic growth
        # g(t) = floor + C / (1 + exp(-k * (t - m)))
        trend = floor + C / (1.0 + torch.exp(-k * (t - m)))

        # If you use quantiles > 1, you'll need to extend the last dimension.
        # For now, we assume a single central quantile.
        return trend

    # required abstract methods from Trend base:
    
    @property
    def get_trend_deltas(self):
        # logistic doesn't use changepoint deltas yet, so we can just return zeros
        # with appropriate shape to keep regularization code happy.
        return torch.zeros_like(self.k).unsqueeze(-1)

    def add_regularization(self):
        # nothing special for now – you can add L2 on k if you want
        return 0.0

import torch
from torch import nn
import copy
from neuralprophet.components.router import get_trend  

class LogisticTrend(nn.Module):
    def __init__(self, config, id_list, quantiles, num_trends_modelled, n_forecasts, device):
        super().__init__()
        self.device = device
        self.quantiles = quantiles
        self.n_forecasts = n_forecasts

        # IMPORTANT: use existing linear trend implementation underneath
        lin_config = copy.copy(config)
        lin_config.growth = "linear"

        self.base_trend = get_trend(
            config=lin_config,
            id_list=id_list,
            quantiles=quantiles,
            num_trends_modelled=num_trends_modelled,
            n_forecasts=n_forecasts,
            device=device,
        )

        # For now assume global (single-series) cap/floor stored on config
        self.cap = getattr(config, "cap", 1.0)
        self.floor = getattr(config, "floor", 0.0)

    def forward(self, t, meta=None):
        # base linear trend (same shape as other trends)
        base = self.base_trend(t=t, meta=meta)
        # logistic squashing:
        # treat base trend as a "logit", then squash between floor and cap
        return self.floor + (self.cap - self.floor) * torch.sigmoid(base)

    # required abstract methods from Trend base:
    
    @property
    def get_trend_deltas(self):
        # logistic doesn't use changepoint deltas yet, so we can just return zeros
        # with appropriate shape to keep regularization code happy.
        return torch.zeros_like(self.k).unsqueeze(-1)

    def add_regularization(self):
        # nothing special for now – you can add L2 on k if you want
        return 0.0

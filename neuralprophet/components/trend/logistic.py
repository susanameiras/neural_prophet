import torch
from torch import nn
import copy

class LogisticTrend(nn.Module):
    def __init__(
        self,
        config,
        id_list,
        quantiles,
        num_trends_modelled,
        n_forecasts,
        device,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.id_list = id_list
        self.id_to_idx = {id_name: i for i, id_name in enumerate(id_list)}

        # --- Read cap/floor from config ---
        cap_dict = config.cap or {}
        floor_dict = config.floor or {}
        
        # store per-ID caps/floors as tensors, default if missing
        caps = []
        floors = []
        for id_name in id_list:
            caps.append(float(cap_dict.get(id_name, 1.0)))
            floors.append(float(floor_dict.get(id_name, 0.0)))

        self.register_buffer(
            "cap",
            torch.tensor(caps, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "floor",
            torch.tensor(floors, dtype=torch.float32, device=device)
        )

        # self.cap = nn.Parameter(torch.tensor(caps, dtype=torch.float32, device=device), requires_grad=False)
        # self.floor = nn.Parameter(torch.tensor(floors, dtype=torch.float32, device=device), requires_grad=False)

        # underlying linear/piecewise-linear trend
        from neuralprophet.components.trend.linear import GlobalLinearTrend, LocalLinearTrend
        from neuralprophet.components.trend.piecewise_linear import GlobalPiecewiseLinearTrend, LocalPiecewiseLinearTrend

        base_trend_cls = GlobalLinearTrend if int(config.n_changepoints) == 0 else GlobalPiecewiseLinearTrend

        # IMPORTANT: pass same config but treat growth as 'linear' internally
        lin_config = copy.copy(config)
        lin_config.growth = "linear"

        self.base_trend = base_trend_cls(
            config=lin_config,
            id_list=id_list,
            quantiles=quantiles,
            num_trends_modelled=num_trends_modelled,
            n_forecasts=n_forecasts,
            device=device,
        )

    def forward(self, t, id_idx=None):
        """
        Parameters
        ----------
        t : torch.Tensor
            normalized time index, shape [batch, n_lags + n_forecasts]
        id_idx : torch.Tensor or None
            indices of IDs in `id_list` for each row in t. Shape [batch]

        Returns
        -------
        trend : torch.Tensor
            logistic trend, shape [batch, n_forecasts] (or similar to base_trend)
        """
        # latent linear trend (unbounded)
        z = self.base_trend(t, id_idx=id_idx)

        # pick per-row cap/floor
        if id_idx is None:
            # single ID case
            cap = self.cap[0]
            floor = self.floor[0]
        else:
            cap = self.cap[id_idx]
            floor = self.floor[id_idx]

        # ensure shapes are broadcastable
        while cap.dim() < z.dim():
            cap = cap.unsqueeze(-1)
            floor = floor.unsqueeze(-1)

        # logistic transform: [floor, cap]
        return floor + torch.sigmoid(z) * (cap - floor)

    # required abstract methods from Trend base:
    
    @property
    def get_trend_deltas(self):
        # logistic doesn't use changepoint deltas yet, so we can just return zeros
        # with appropriate shape to keep regularization code happy.
        return torch.zeros_like(self.k).unsqueeze(-1)

    def add_regularization(self):
        # nothing special for now – you can add L2 on k if you want
        return 0.0

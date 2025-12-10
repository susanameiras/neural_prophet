from typing import List, Dict, Optional
import copy

import torch
from torch import nn

from neuralprophet.components.trend import Trend
from neuralprophet.components.trend.linear import GlobalLinearTrend, LocalLinearTrend
from neuralprophet.components.trend.piecewise_linear import GlobalPiecewiseLinearTrend, LocalPiecewiseLinearTrend

class LogisticTrend(Trend):
    """
    Logistic trend wrapper around the existing linear / piecewise-linear trends.

    We first compute an unconstrained linear trend g(t) using the standard
    trend classes (global/local, with/without changepoints), and then squash
    it through a logistic function between floor and cap:

        y(t) = floor + (cap - floor) * sigmoid(g(t))

    where floor and cap are taken per time-series ID from config.cap / config.floor.

    Expected config attributes:
        - config.growth == "logistic"
        - optional: config.cap: Dict[str, float]  (per ID)
        - optional: config.floor: Dict[str, float]  (per ID)
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
        # Initialize base Trend (sets device, id_list, etc.)
        super().__init__(config, id_list, quantiles, num_trends_modelled, n_forecasts, device)

        # Store per-ID caps / floors from config, if present
        # (config_trend should be constructed with cap=cap_dict, floor=floor_dict)
        self.cap_dict: Optional[Dict[str, float]] = getattr(config, "cap", None)
        self.floor_dict: Optional[Dict[str, float]] = getattr(config, "floor", None)

        # Build an internal *linear* trend module to produce the unconstrained logit g(t)
        lin_config = copy.copy(config)
        lin_config.growth = "linear"   # important: treat internal trend as linear

        args = dict(
            config=lin_config,
            id_list=id_list,
            quantiles=quantiles,
            num_trends_modelled=num_trends_modelled,
            n_forecasts=n_forecasts,
            device=device,
        )

        # This logic mirrors router.get_trend(), but we do it locally here
        if num_trends_modelled == 1:
            # global trend
            if int(config.n_changepoints) == 0:
                self._base_trend = GlobalLinearTrend(**args)
            else:
                self._base_trend = GlobalPiecewiseLinearTrend(**args)
        else:
            # local trend per ID
            if int(config.n_changepoints) == 0:
                self._base_trend = LocalLinearTrend(**args)
            else:
                self._base_trend = LocalPiecewiseLinearTrend(**args)

    # --------- helper: get per-sample cap / floor tensors ---------

    def _get_cap_floor(self, meta):
        """
        Builds cap and floor tensors of shape (batch, 1, 1) from
        self.cap_dict / self.floor_dict and meta["df_name"].
        """
        if meta is None or "df_name" not in meta or self.cap_dict is None:
            return None, None

        df_names = meta["df_name"]  # list[str], len = batch_size
        device = self._base_trend.device

        caps = torch.tensor(
            [self.cap_dict.get(name) for name in df_names],
            dtype=torch.float32,
            device=device,
        ).view(-1, 1, 1)

        if self.floor_dict is not None:
            floors = torch.tensor(
                [self.floor_dict.get(name) for name in df_names],
                dtype=torch.float32,
                device=device,
            ).view(-1, 1, 1)
        else:
            floors = torch.zeros_like(caps)

        return caps, floors

    # --------- required Trend API ---------

    def forward(self, t: torch.Tensor, meta=None) -> torch.Tensor:
        """
        Compute logistic trend.

        Parameters
        ----------
        t : torch.Tensor
            Normalized time, shape (batch, n_forecasts)
        meta : dict
            Contains 'df_name': list of time series IDs, len = batch

        Returns
        -------
        torch.Tensor
            Trend component, shape (batch, n_forecasts, n_quantiles)
        """
        # Unconstrained linear or piecewise-linear trend g(t)
        lin_trend = self._base_trend(t, meta=meta)

        caps, floors = self._get_cap_floor(meta)
        if caps is None:
            # No cap/floor information -> fall back to linear trend
            return lin_trend

        # Broadcast caps / floors to match lin_trend
        # lin_trend: (B, T, Q)
        # caps/floors: (B, 1, 1)
        # -> PyTorch broadcasting handles it
        return floors + (caps - floors) * torch.sigmoid(lin_trend)

    @property
    def get_trend_deltas(self):
        """
        Delegate regularization deltas to the internal linear trend.
        """
        return self._base_trend.get_trend_deltas

    def add_regularization(self):
        """
        Delegate regularization computation to the internal linear trend.
        """
        return self._base_trend.add_regularization()

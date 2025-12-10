import torch
import torch.nn as nn

from neuralprophet.components.trend import Trend
from neuralprophet.utils_torch import init_parameter


class LogisticTrend(Trend):
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

    @property
    def get_trend_deltas(self):
        """trend deltas for regularization.

        update if trend is modelled differently"""
        if self.config_trend is None:
            trend_delta = None
        else:
            trend_delta = self.trend_deltas

        return trend_delta

    def add_regularization(self):
        pass

from __future__ import annotations

import torch
import torch.nn as nn

from neuralprophet.components.trend import Trend
from neuralprophet.utils_torch import init_parameter


class LogisticTrend(Trend):
    """Base class for logistic trend components.

    A logistic trend has the functional form

    .. math::

        y(t) = \mathrm{bias} + \mathrm{cap} \times \sigma\bigl(k\, (t - m)\bigr),

    where :math:`\sigma(x) = 1/(1+\exp(-x))` is the sigmoid function,
    ``cap`` is a capacity parameter, ``k`` controls the steepness of the curve
    and ``m`` shifts the curve along the time axis. The ``bias`` term is
    inherited from :class:`neuralprophet.components.trend.base.Trend` and allows
    the entire curve to be shifted up or down.

    Parameters are initialised using Xavier normal initialisation via
    :func:`neuralprophet.utils_torch.init_parameter`.
    """

    def __init__(
        self,
        config,
        id_list,
        quantiles,
        num_trends_modelled,
        n_forecasts,
        device,
    ) -> None:
        super().__init__(
            config=config,
            n_forecasts=n_forecasts,
            num_trends_modelled=num_trends_modelled,
            quantiles=quantiles,
            id_list=id_list,
            device=device,
        )
        # Base growth rate k0 for each quantile and (potentially) each time series.
        # Dimensions: [n_quantiles, num_trends_modelled, 1]
        self.trend_k0 = init_parameter(
            dims=[len(self.quantiles), self.num_trends_modelled, 1]
        )
        # Capacity parameter controlling saturation. A soft absolute is applied in
        # forward to ensure positivity. Dimensions match trend_k0.
        self.trend_cap = init_parameter(
            dims=[len(self.quantiles), self.num_trends_modelled, 1]
        )
        # Horizontal shift of the logistic curve. Dimensions match trend_k0.
        self.trend_m = init_parameter(
            dims=[len(self.quantiles), self.num_trends_modelled, 1]
        )

    @property
    def get_trend_deltas(self):
        """Return trend deltas for regularisation.

        Logistic trend currently does not implement changepoints, so no deltas
        are returned. Returning ``None`` disables changepoint regularisation.
        """

        return None

    def add_regularization(self) -> None:
        """Add optional regularisation to the logistic parameters.

        This method is left as a placeholder so that future extensions can
        implement regularisation (e.g. penalising overly large capacities or
        growth rates). Currently no additional loss terms are added.
        """

        pass


class GlobalLogisticTrend(LogisticTrend):
    """Logistic trend shared across all time series."""

    def __init__(
        self,
        config,
        id_list,
        quantiles,
        num_trends_modelled,
        n_forecasts,
        device,
    ) -> None:
        super().__init__(
            config=config,
            id_list=id_list,
            quantiles=quantiles,
            num_trends_modelled=num_trends_modelled,
            n_forecasts=n_forecasts,
            device=device,
        )

    def forward(self, t: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Evaluate the logistic trend for a batch of times.

        Parameters
        ----------
        t : torch.Tensor, float
            Normalised time variable with shape ``(batch_size, n_forecasts)``.
        meta : torch.Tensor
            Metadata containing the time series identifier for each sample. This
            argument is unused for global trends but maintained for signature
            compatibility.

        Returns
        -------
        torch.Tensor
            Logistic trend of shape ``(batch_size, n_forecasts, n_quantiles)``.
        """
        # Extract parameter tensors and permute to align dimensions with input
        # rate, cap and m have shape [num_trends_modelled, 1, n_quantiles]
        rate = self.trend_k0.permute(1, 2, 0)
        cap = self.trend_cap.permute(1, 2, 0)
        m = self.trend_m.permute(1, 2, 0)
        # Ensure capacity is positive by taking the absolute value. This avoids
        # negative saturation which would invert the curve.
        cap = torch.abs(cap)
        # Broadcast time tensor to shape [batch_size, n_forecasts, 1]
        t_unsqueezed = t.unsqueeze(dim=2)
        # Compute logistic curve: cap * sigmoid(rate * (t - m))
        trend = cap * torch.sigmoid(rate * (t_unsqueezed - m))
        # Add learnable bias and broadcast to match dimensions
        return self.bias.unsqueeze(dim=0).unsqueeze(dim=0) + trend


class LocalLogisticTrend(LogisticTrend):
    """Logistic trend with separate parameters for each time series."""

    def __init__(
        self,
        config,
        id_list,
        quantiles,
        num_trends_modelled,
        n_forecasts,
        device,
    ) -> None:
        super().__init__(
            config=config,
            id_list=id_list,
            quantiles=quantiles,
            num_trends_modelled=num_trends_modelled,
            n_forecasts=n_forecasts,
            device=device,
        )

    def forward(self, t: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Evaluate the logistic trend for a batch of times with local parameters.

        Parameters
        ----------
        t : torch.Tensor, float
            Normalised time variable with shape ``(batch_size, n_forecasts)``.
        meta : torch.Tensor
            Integer tensor of shape ``(batch_size,)`` identifying the time series
            each sample belongs to. This is converted into a one‑hot encoding
            to select the appropriate trend parameters.

        Returns
        -------
        torch.Tensor
            Logistic trend of shape ``(batch_size, n_forecasts, n_quantiles)``.
        """
        # Convert meta indices to one‑hot encoding: shape [batch_size, n_series]
        meta_one_hot = nn.functional.one_hot(meta, num_classes=len(self.id_list))
        # Select parameters corresponding to each sample's time series. The
        # unsqueeze operations align dimensions so that broadcasting works when
        # summing over the id dimension. The resulting tensors have shape
        # [batch_size, 1, n_quantiles].
        rate = (
            torch.sum(
                meta_one_hot.unsqueeze(dim=0).unsqueeze(dim=-1) * self.trend_k0.unsqueeze(dim=1),
                dim=2,
            )
        ).permute(1, 2, 0)
        cap = (
            torch.sum(
                meta_one_hot.unsqueeze(dim=0).unsqueeze(dim=-1) * self.trend_cap.unsqueeze(dim=1),
                dim=2,
            )
        ).permute(1, 2, 0)
        m = (
            torch.sum(
                meta_one_hot.unsqueeze(dim=0).unsqueeze(dim=-1) * self.trend_m.unsqueeze(dim=1),
                dim=2,
            )
        ).permute(1, 2, 0)
        # Enforce positive capacity
        cap = torch.abs(cap)
        # Broadcast time tensor to match parameter dimensions
        t_unsqueezed = t.unsqueeze(dim=2)
        # Compute logistic curve for each sample
        trend = cap * torch.sigmoid(rate * (t_unsqueezed - m))
        # Add bias and return
        return self.bias.unsqueeze(dim=0).unsqueeze(dim=0) + trend

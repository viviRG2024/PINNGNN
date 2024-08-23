from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def l1_loss(prediction: torch.Tensor, target: torch._tensor, size_average: Optional[bool] = None, reduce: Optional[bool] = None, reduction: str = "mean") -> torch.Tensor:
    """unmasked mae."""

    return F.l1_loss(prediction, target, size_average=size_average, reduce=reduce, reduction=reduction)


def l2_loss(prediction: torch.Tensor, target: torch.Tensor, size_average: Optional[bool] = None, reduce: Optional[bool] = None, reduction: str = "mean") -> torch.Tensor:
    """unmasked mse"""

    return F.mse_loss(prediction, target, size_average=size_average, reduce=reduce, reduction=reduction)


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(prediction-target)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean squared error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean squared error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (prediction-target)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """root mean squared error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value . Defaults to np.nan.

    Returns:
        torch.Tensor: root mean squared error
    """

    return torch.sqrt(masked_mse(prediction=prediction, target=target, null_val=null_val))


def masked_mape(prediction: torch.Tensor, target: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Masked mean absolute percentage error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value.
                                    In the mape metric, null_val is set to 0.0 by all default.
                                    We keep this parameter for consistency, but we do not allow it to be changed.

    Returns:
        torch.Tensor: masked mean absolute percentage error
    """
    # we do not allow null_val to be changed
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values
    target = torch.where(torch.abs(target) < 1e-4, torch.zeros_like(target), target)
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.abs(prediction-target)/target)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def pinngnn_physics_loss(prediction: torch.Tensor, target: torch.Tensor, occupy: torch.Tensor, speed: torch.Tensor, dt: float = 1, dx: float = 1, v_f: float = 30, rho_max: float = 100) -> torch.Tensor:
    """
    Custom loss function combining traditional loss with physics-based loss.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): target values
        occupy (torch.Tensor): occupancy values
        speed (torch.Tensor): speed values
        dt (float): time step
        dx (float): space step
        v_f (float): free flow speed
        rho_max (float): maximum density

    Returns:
        torch.Tensor: combined loss
    """
    # Traditional loss (e.g., MSE)
    traditional_loss = F.mse_loss(prediction, target)

    # Physics-based loss components
    rho = occupy / 100  # Convert occupancy to density
    q = prediction

    # 1. Conservation equation loss
    continuity_loss = torch.mean(torch.abs(
        (rho[:, 1:, :] - rho[:, :-1, :]) / dt + 
        (q[:, :, 1:] - q[:, :, :-1]) / dx
    ))

    # 2. Fundamental diagram relationship loss
    fundamental_diagram_loss = torch.mean((q - rho * speed)**2)

    # 3. Speed-density relationship loss
    speed_density_loss = torch.mean((speed - v_f * (1 - rho / rho_max))**2)

    # 4. Temporal consistency loss
    temporal_consistency_loss = torch.mean(
        (rho[:, 1:, :] - rho[:, :-1, :])**2 + 
        (speed[:, 1:, :] - speed[:, :-1, :])**2 + 
        (q[:, 1:, :] - q[:, :-1, :])**2
    )

    # Combine physics losses
    physics_loss = (
        0.1 * continuity_loss + 
        0.1 * fundamental_diagram_loss + 
        0.1 * speed_density_loss + 
        0.1 * temporal_consistency_loss
    )

    # Combine traditional and physics losses
    total_loss = traditional_loss + physics_loss

    return total_loss
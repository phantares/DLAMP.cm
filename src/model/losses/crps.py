import torch
import torch.nn as nn
from torch.distributions import Normal


class CRPSLoss(nn.Module):

    def __init__(self, reduction: str = "mean", mode: str = "norm"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        assert mode in ("regress", "norm")

        self.reduction = reduction
        self.mode = mode

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        match self.mode:
            case "regress":
                return nn.L1Loss(reduction=self.reduction)(input, target)

            case "norm":
                C = input.size(-4)
                assert C % 2 == 0

                mu = input[..., : C // 2, :, :, :]
                sigma = torch.exp(input[..., C // 2 :, :, :, :])
                standard_normal = Normal(torch.zeros_like(mu), torch.ones_like(mu))

                z = (target - mu) / sigma
                phi_z = standard_normal.log_prob(z).exp()  # PDF
                Phi_z = standard_normal.cdf(z)  # CDF

                crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / torch.pi**0.5)

        if self.reduction == "mean":
            return crps.mean()
        elif self.reduction == "sum":
            return crps.sum()
        return crps  # "none"

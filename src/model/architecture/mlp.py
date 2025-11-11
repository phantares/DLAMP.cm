import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_channel,
        hidden_channel,
        out_channel,
        dropout_rate=0.0,
    ) -> None:
        """
        Args:
            in_channel (int): Number of input channels.
            hidden_channel (int): Number of hidden channels.
            out_channel (int): Number of output channels.
            dropout_rate (float, optional): Output dropout rate. Default: 0.0
        """

        super().__init__()

        self.linear1 = nn.Linear(in_channel, hidden_channel)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_channel, out_channel)
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)

        return x

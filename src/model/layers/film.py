import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, film_channel, channel):
        super().__init__()

        self.gb = nn.Linear(film_channel, 2 * channel)

    def forward(self, x, film_base):
        gamma, beta = torch.chunk(self.gb(film_base), 2, dim=-1)

        return x * (1 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]

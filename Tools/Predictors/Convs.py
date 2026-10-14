import torch
from torch import nn

import sys
sys.path.insert(0, '/home/wangyucheng/projects/Benchmark/Tools/Predict_models/Seq2Func_pytorch/Custom_Models')
from Conv_elements import *


class ConvModel1(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()

        mid_dim = dim // 2

        self.stem = nn.Sequential(
            # Conv1d(input_dim → dim/2)
            nn.Conv1d(
                input_dim,
                mid_dim,
                kernel_size=9,
                stride=1,
                padding=4, 
            ),
            # Residual block
            ConvBlock1(
                    mid_dim,
                    mid_dim,
                    kernel_size=1,
                    stride=1,
                    padding="same",
                    residual=True
            ),
            # pooling
            MaxPool1D(
                kernel_size=2,
                stride=2,
                ceil_mode=True
            )
        )

        ################ conv tower ################
        blocks = []
        current_dim = mid_dim
        for _ in range(6):
            next_dim = current_dim + 128
            # Conv1d
            blocks.append(
                ConvBlock1(
                    current_dim,
                    next_dim,
                    kernel_size=3,
                    stride=1,
                    padding="same",
                    residual=False
                )
            )
            # Residual block
            blocks.append(
                ConvBlock1(
                    next_dim,
                    next_dim,
                    kernel_size=1,
                    stride=1,
                    padding="same",
                    residual=True
                )
            )
            # pooling
            blocks.append(
                MaxPool1D(
                    kernel_size=2,
                    stride=2,
                    ceil_mode=True
                )
            )

            current_dim = next_dim

        self.conv_tower = nn.Sequential(*blocks)


    def forward(self, x):
        x = self.stem(x)
        x = self.conv_tower(x)
        return x

from torch import nn

import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from featureSelectionLayer import FeatureSelectionV2

class LiverModel(nn.Module):
        def __init__(self, name: str = ''):
          super().__init__()
          self.name=name
          self.block_1=nn.Sequential(
              nn.Linear(22277, 512),
              nn.BatchNorm1d(512),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(512, 128),
              nn.BatchNorm1d(128),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(128, 50),
              nn.BatchNorm1d(50),
              nn.ReLU(),
              nn.Linear(50, 1)
          )
        def forward(self, x):
          return self.block_1(x)

class LiverModelWithFSL(nn.Module):
        def __init__(self, name: str = ''):
          super().__init__()
          self.name = name
          self.block_1=nn.Sequential(
              FeatureSelectionV2(22277),
              nn.Linear(22277, 512),
              nn.BatchNorm1d(512),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(512, 128),
              nn.BatchNorm1d(128),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(128, 50),
              nn.BatchNorm1d(50),
              nn.ReLU(),
              nn.Linear(50, 1)
          )
        def forward(self, x):
          return self.block_1(x)
from torch import nn

import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from featureSelectionLayer import FeatureSelectionV2

class BreastModel(nn.Module):
        def __init__(self, name: str = ''):
          super().__init__()
          self.name=name
          self.block_1=nn.Sequential(
              nn.Linear(54675, 100),
              nn.BatchNorm1d(100),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(100, 200),
              nn.BatchNorm1d(200),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(200, 100),
              nn.BatchNorm1d(100),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(100, 100),
              nn.BatchNorm1d(100),
              nn.ReLU(),
              nn.Linear(100, 6)
          )
        def forward(self, x):
          return self.block_1(x)

class BreastModelModelWithFSL(nn.Module):
        def __init__(self, name: str = ''):
          super().__init__()
          self.name = name
          self.block_1=nn.Sequential(
              FeatureSelectionV2(54675),
              nn.Linear(54675, 100),
              nn.BatchNorm1d(100),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(100, 200),
              nn.BatchNorm1d(200),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(200, 100),
              nn.BatchNorm1d(100),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(100, 100),
              nn.BatchNorm1d(100),
              nn.ReLU(),
              nn.Linear(100, 6)
          )
        def forward(self, x):
          return self.block_1(x)
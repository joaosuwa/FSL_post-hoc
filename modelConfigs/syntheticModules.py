from torch import nn

import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from featureSelectionLayer import FeatureSelectionV2

class SynthModel(nn.Module):
        def __init__(self, name: str = ''):
          super().__init__()
          self.name = name
          self.block_1=nn.Sequential(
              nn.Linear(100, 128),
              nn.BatchNorm1d(128),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(128, 64),
              nn.BatchNorm1d(64),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(64, 16),
              nn.BatchNorm1d(16),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(16, 1),
          )
        def forward(self, x):
          return self.block_1(x)
        
class SynthModelWithFSL(nn.Module):
        def __init__(self, name: str = ''):
          super().__init__()
          self.name = name
          self.block_1=nn.Sequential(
              FeatureSelectionV2(100),
              nn.Linear(100, 128),
              nn.BatchNorm1d(128),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(128, 64),
              nn.BatchNorm1d(64),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(64, 16),
              nn.BatchNorm1d(16),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(16, 1),
          )
        def forward(self, x):
          return self.block_1(x)
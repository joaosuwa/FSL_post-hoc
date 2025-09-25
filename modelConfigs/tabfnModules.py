from tabpfn import TabPFNClassifier
from torch import nn
import torch
import numpy as np
from featureSelectionLayer import FeatureSelectionV2
device = "cuda" if torch.cuda.is_available() else "cpu"


class TabPFNForSHAP(nn.Module):
    def __init__(self, tabPFN: TabPFNClassifier, name: str = ''):
        super().__init__()
        self.name = name
        self.tabPFN = tabPFN
        self.tabPFN_model = tabPFN.model_
        for param in self.tabPFN_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = torch.from_numpy(x).to(device)
        output = self.tabPFN.forward(x, return_logits=True)
        output = output.permute(0, 2, 1).squeeze().detach().cpu().numpy()
        output = np.array([output]) if x.shape[0] == 1 else output
        return output
    

class TabPFNModelWithFSL(nn.Module):
    def __init__(self, tabPFN: TabPFNClassifier, name: str = ''):
        super().__init__()
        self.name = name
        self.fs = FeatureSelectionV2(100)
        self.tabPFN = tabPFN
        self.tabPFN_model = tabPFN.model_
        for param in self.tabPFN_model.parameters():
            param.requires_grad = False


    def forward(self, x):
        weighted_x = self.fs(x)
        output = self.tabPFN.forward(weighted_x, return_logits=True)
        return output
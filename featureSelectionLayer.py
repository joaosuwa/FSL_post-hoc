import torch
from torch import nn
from collections import OrderedDict
device = "cuda" if torch.cuda.is_available() else "cpu"

## Feature Selection Layer

class FeatureSelectionV2(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        initial_weight = torch.ones(1, in_features).to(device)
        self.weight = nn.Parameter(initial_weight)
        self.activation = nn.ReLU()

    def forward(self, input):
        weighted_input = input * self.get_activated_weights()
        return weighted_input

    def get_weights(self):
        return self.weight

    def get_activated_weights(self):
        return self.activation(self.weight)
    
def fs_layer_regularization(model, l=0.001):
    return l * torch.sum(torch.abs(model.block_1[0].get_weights())) # L1 regularzation

def transfer_weights(model_v1, model_v2):
    state_dict_v1 = model_v1.state_dict()
    state_dict_v2 = model_v2.state_dict()

    new_state_dict_v2 = OrderedDict()

    for i, (key_v2, param_v2) in enumerate(state_dict_v2.items()):
        # Pular parâmetros da FeatureSelectionV2 (primeiro layer do V2)
        if key_v2.startswith("block_1.0."):
            new_state_dict_v2[key_v2] = param_v2  # mantém como está (FeatureSelection)
            continue

        # Calcular o índice da camada correspondente em V1 (offset -1 por causa da FeatureSelection)
        key_index = int(key_v2.split('.')[1])
        key_v1_index = key_index - 1
        key_v1 = key_v2.replace(f"block_1.{key_index}.", f"block_1.{key_v1_index}.")

        # Copiar valor correspondente do V1
        if key_v1 in state_dict_v1:
            new_state_dict_v2[key_v2] = state_dict_v1[key_v1]
        else:
            print(f"Aviso: {key_v1} não encontrado em V1. Mantendo valor atual.")
            new_state_dict_v2[key_v2] = param_v2  # fallback

    # Carregar os pesos modificados no modelo V2
    model_v2.load_state_dict(new_state_dict_v2)

def freezeParams(model):
    # Congelar todas as camadas exceto FeatureSelectionV2
    for name, param in model.named_parameters():
        if 'block_1.0' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for name, param in model.named_parameters():
        print(name, param.requires_grad)


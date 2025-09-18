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
        return input * self.get_activated_weights()

    def get_weights(self):
        return self.weight

    def get_activated_weights(self):
        return self.activation(self.weight)
    
class TabNetFeatureSelectionV2(nn.Module):
    def __init__(self, in_features, internal_model, name='', freeze_internal_model=True):
        super().__init__()
        self.name = name
        initial_weight = 1.0 * torch.ones(1, in_features).to(device)
        self.weight = nn.Parameter(initial_weight)
        self.activation = nn.ReLU()
        self.internal_model = internal_model
        self.block_1 = [self]
        for param in self.internal_model.parameters():
            param.requires_grad = not freeze_internal_model

    def forward(self, input):
        weighted_input = input * self.get_activated_weights()
        output, M_loss = self.internal_model.forward(weighted_input)
        return output

    def get_weights(self):
        return self.weight

    def get_activated_weights(self):
        return self.activation(self.weight)
    
    def train(self, value=True):
        super().train(value)
        self.internal_model.train(value)

def fs_layer_regularization(model, l=0.001):
    # L1 regularzation
    return l * torch.sum(torch.abs(model.block_1[0].get_weights()))

def transfer_weights(model_v1, model_v2):
    state_dict_v1 = model_v1.state_dict()
    state_dict_v2 = model_v2.state_dict()

    new_state_dict_v2 = OrderedDict()

    for _, (key_v2, param_v2) in enumerate(state_dict_v2.items()):
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
            raise ValueError(f"Erro na transferência: {key_v1} não encontrado em V1. Mantendo valor atual.")

    # Carregar os pesos modificados no modelo V2
    model_v2.load_state_dict(new_state_dict_v2)
    model_v2.to(device)

def freezeParams(model):
    # Congelar todas as camadas exceto FeatureSelectionV2
    for name, param in model.named_parameters():
        if 'block_1.0' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

def compare_model_weights(model1, model2, logger):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    for key_v2, _ in state_dict2.items():
        if key_v2.startswith("block_1.0."):
            continue

        key_index = int(key_v2.split('.')[1])
        key_v1_index = key_index - 1
        key_v1 = key_v2.replace(f"block_1.{key_index}.", f"block_1.{key_v1_index}.")

        weights_equal = torch.equal(state_dict1[key_v1], state_dict2[key_v2])
        logger.log_text(f"Layer '{key_v1}' and '{key_v2}': {'✅ Same' if weights_equal else '❌ Different'}")

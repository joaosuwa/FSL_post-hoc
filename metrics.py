import torch
import pandas as pd
from sklearn.metrics import silhouette_score
from utils import find_normalized_weights

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

def Selection_Accuracy(rank_features, informative_features, n, feature_cols):
  informative_features_selected = []

  rank_series = pd.Series(rank_features.numpy(), index=feature_cols)

  top_features = rank_series.sort_values(ascending=False).head(n).index

  for feature in top_features:
    if feature in informative_features:
      informative_features_selected.append(feature)

  PIFS = len(informative_features_selected) / len(informative_features)
  PSFI = len(informative_features_selected) / len(top_features)

  #print(f'Avaliação dos top {n} features: \n')
  #print(f'PIFS: {PIFS}')
  #print(f'PSFI: {PSFI}\n')

  return PIFS, PSFI

def silhouetteMetric(X_tsne, y):
    # Sihouette Coeficient after applying t-SNE:
    silhouette_avg_tsne = silhouette_score(X_tsne, y)
    print(f"Silhouette Score after applying t-SNE: {silhouette_avg_tsne}")

def informativeFeaturesMetric(model, feature_cols):

    informative_features = [col for col in feature_cols if 'informative' in col]
    n_features = len(feature_cols)

    # get the ranked list for all the features
    normalized_weights_feature_selection = find_normalized_weights(model).squeeze().detach().cpu()

    if len(informative_features) == 0:
        return []

    scores = []

    for selection_size in range(1, n_features + 1):

        # calculate the PIFS and PSFI for all the features subsets from more significant to less significant
        PIFS, PSFI = Selection_Accuracy(normalized_weights_feature_selection, informative_features, selection_size, feature_cols)
        dict = {'PIFS': float(f"{PIFS:.3f}"), 'PSFI': float(f"{PSFI:.3f}")}
        scores.append(dict)

    return scores
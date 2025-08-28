import pandas as pd
from IPython.display import display

def find_normalized_weights(model):
    weights_feature_selection = model.block_1[0].get_weights()
    mean_global = weights_feature_selection.mean()
    std_global = weights_feature_selection.std()
    epsilon = 1e-8
    normalized_weights_feature_selection = (weights_feature_selection - mean_global) / (std_global + epsilon)
    return normalized_weights_feature_selection

def displayTopFeatures(model, feature_cols):

    normalized_weights_feature_selection = find_normalized_weights(model)

    weights_df = pd.DataFrame({
        'feature': feature_cols,
        'normalized_weight': normalized_weights_feature_selection.squeeze().detach().cpu().numpy()
    })

    sorted_weights_df = weights_df.sort_values(by='normalized_weight', ascending=False)

    print("Pesos normalizados por feature (ordenado):")
    display(sorted_weights_df)

    sorted_weights_df['abs_weight'] = sorted_weights_df['normalized_weight']
    top_features = sorted_weights_df.sort_values(by='abs_weight', ascending=False).head(30)
    print("\nTop 30 Features por valor do peso normalizado:")
    display(top_features)

def min_max_normalized_weights(model): # Used to get the min_max normalized values of the feature selection
    
    weights = model.block_1[0].get_activated_weights()
    epsilon = 1e-8
    min = weights.min()
    max = weights.max()
    min_max_normalized_weights_feature_selection = (weights - min) / (max - min + epsilon)

    return min_max_normalized_weights_feature_selection
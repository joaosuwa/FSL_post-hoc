import os
import uuid
import pandas as pd
import numpy as np
from IPython.display import display
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
import torch


def calculate_kruskal_dunn(without_fsl_results, with_fsl_results, with_posthoc_fsl_results, p_value_threshold=0.01):
    result = ""
    try:
        _, p_value = kruskal(without_fsl_results, with_fsl_results, with_posthoc_fsl_results)
    except ValueError as e:
        if "All numbers are identical" in str(e):
            p_value = 1.0  # No difference
        else:
            raise
    if p_value < p_value_threshold:
        result += "Significant difference detected (p < 0.01). "
        data = without_fsl_results + with_fsl_results + with_posthoc_fsl_results
        groups = ['without_weights'] * len(without_fsl_results) + ['with_fsl'] * len(with_fsl_results) + ['with_posthoc_fsl'] * len(with_posthoc_fsl_results)
        df = pd.DataFrame({'value': data, 'group': groups})
        # Dunn’s test with Bonferroni correction
        dunn_results = posthoc_dunn(df, val_col='value', group_col='group', p_adjust='bonferroni')
        result += f"Dunn's post-hoc test results (p-values):\n {dunn_results}"
    else:
        result = "No significant difference (p ≥ 0.01)."
    return result

def get_feature_rankings(model, feature_columns):
    weights = get_feature_weights_as_numpy(model)
    ordered_indices = weights.argsort()[::-1]
    return [feature_columns[i] for i in ordered_indices]

def get_feature_weights_as_numpy(model):
    weights = model.block_1[0].get_weights()
    return weights.squeeze().detach().cpu().numpy()

def find_normalized_weights(model):
    weights = model.block_1[0].get_activated_weights()
    epsilon = 1e-8
    min = torch.min(weights)
    max = torch.max(weights)
    return (weights - min) / (max - min + epsilon)

def displayTopFeatures(model, feature_cols, print_function=print, display_function=display):
    feature_weights = get_feature_weights_as_numpy(model)
    normalized_feature_weights = find_normalized_weights(model)

    weights_df = pd.DataFrame({
        'feature': feature_cols,
        'weight': feature_weights,
        'normalized_weight': normalized_feature_weights.squeeze().detach().cpu().numpy()
    })

    sorted_weights_df = weights_df.sort_values(by='normalized_weight', ascending=False)

    #sorted_weights_df.set_option('display.float_format', lambda x: '%.3f' % x)

    print_function("Pesos normalizados por feature (ordenado):")
    display_function(sorted_weights_df)

    sorted_weights_df['abs_weight'] = sorted_weights_df['normalized_weight']
    top_features = sorted_weights_df.sort_values(by='abs_weight', ascending=False).head(30)
    print_function("\nTop 30 Features por valor do peso normalizado:")

    #top_features.set_option('display.float_format', lambda x: '%.3f' % x)
    display_function(top_features)

def persist_wtsne_input(model, feature_cols, name, logger):
    feature_weights = get_feature_weights_as_numpy(model)

    weights_df = pd.DataFrame({
        'feature': feature_cols,
        'value': feature_weights
    })

    weights_df.to_csv(os.path.join(logger.dir_path, f"{name}-wtsne-input.csv"), index=False, encoding='utf-8', float_format="%.10f")

def min_max_normalized_weights(model): # Used to get the min_max normalized values of the feature selection
    
    weights = model.block_1[0].get_activated_weights()
    epsilon = 1e-8
    min = weights.min()
    max = weights.max()
    min_max_normalized_weights_feature_selection = (weights - min) / (max - min + epsilon)

    return min_max_normalized_weights_feature_selection

def generate_execution_id(name, base_path="results", external_id="", index=None, persist=True):
    execution_id = str(uuid.uuid4())
    if index is not None:
        execution_id = f"{str(index)}_{execution_id}"

    if persist:
        if external_id != "":
            execution_dir = os.path.join(base_path, name, external_id, execution_id)
        else:
            execution_dir = os.path.join(base_path, name, execution_id)
        os.makedirs(execution_dir, exist_ok=True)
    return execution_id

class LogPrinter:
    def __init__(self, name, execution_id, base_path="results", persist=True, father=None):
        self.dir_path = father.dir_path if father is not None else f"{base_path}/{name}"
        self.dir_path = f"{self.dir_path}/{execution_id}"
        self.persist = persist

    def log_text(self, message, filename="log.txt"):
        print(message + "\n")
        if self.persist:
            with open(os.path.join(self.dir_path, filename), "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def log_dataframe(self, dataframe, filename="output.txt"):
        display(dataframe)
        if self.persist:
            pd.options.display.float_format = '{:.10f}'.format
            with open(os.path.join(self.dir_path, filename), "a", encoding="utf-8") as f:
                f.write(dataframe.to_string() + "\n")

    def log_np_array(self, np_array, filename, fmt):
        display(np_array)
        if self.persist:
            np.savetxt(os.path.join(self.dir_path, filename), np_array, fmt=fmt)
            
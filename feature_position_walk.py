import matplotlib.pyplot as plt
import numpy as np


def generate_feature_position_walk_plot(logger, feature_weights_results, feature_columns, model_name, informative_features=None, top_k_threshold=50, y_limit=5000):
    feature_positions = {feature: [] for feature in feature_columns}
    aggregated_feature_positions = {feature: 0 for feature in feature_columns}
    top_k_threshold = min(top_k_threshold, len(feature_columns))

    if informative_features is not None and len(informative_features) > 0:
        feature_weights_results = reorder_executions_by_top_k_similarity(
            feature_weights_results, 
            feature_columns, 
            len(informative_features) if len(informative_features) > 0 else len(feature_columns)
        )

    for feature_weights in feature_weights_results:
        ranking = [feature_columns[i] for i in np.argsort(feature_weights)[::-1]]
        for rank, feature in enumerate(ranking):
            feature_positions[feature].append(rank + 1)
            aggregated_feature_positions[feature] += rank

    filtered_features = sorted(
        [feature for feature, count in aggregated_feature_positions.items() if count > 0],
        key=lambda f: -aggregated_feature_positions[f]
    )#[:(2*y_limit)]

    execution_indices = list(range(1, len(feature_weights_results) + 1))

    # Version WITH legend
    plt.figure(figsize=(14, 10))
    for feature in filtered_features:
        plt.plot(execution_indices, feature_positions[feature], label=feature)

    plt.xlabel("Execution Index")
    plt.ylabel("Ranking Position (1 = Most Important)")
    plt.title("Feature Importance Trajectories Across Executions")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(y_limit, 0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=1)
    plt.tight_layout()
    plt.savefig(f'{logger.dir_path}/feature_position_walk_{model_name}_with_legend.pdf')
    plt.close()

    # Version WITHOUT legend
    plt.figure(figsize=(14, 10))
    for feature, positions in feature_positions.items():
        plt.plot(execution_indices, positions)

    plt.xlabel("Execution Index")
    plt.ylabel("Ranking Position (1 = Most Important)")
    plt.title("Feature Importance Trajectories Across Executions")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(y_limit, 0) 
    plt.tight_layout()
    plt.savefig(f'{logger.dir_path}/feature_position_walk_{model_name}_no_legend.pdf')
    plt.close()


def reorder_executions_by_top_k_similarity(feature_weights_results, feature_columns, k):
    ranking_list = []
    for _, feature_weights in enumerate(feature_weights_results):
        ranking = [feature_columns[i] for i in np.argsort(feature_weights)[::-1]]
        ranking_list.append(ranking)

    reference = ranking_list[0][:k]

    def similarity_score(exec):
        return sum(1 for i in range(2) if exec[i] == reference[i])

    scored = [(i, similarity_score(exec)) for i, exec in enumerate(ranking_list)]
    sorted_indices = [i for i, _ in sorted(scored, key=lambda x: -x[1])]
    reordered = [feature_weights_results[i] for i in sorted_indices]
    return reordered

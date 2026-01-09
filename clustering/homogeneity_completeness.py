import numpy as np
from typing import Dict
import json
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from collections import Counter
import pandas as pd

def evaluate_homogeneity_completeness(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    ignore_noise: bool = True,
    save_path = None
) -> Dict[str, float]:
    """
    Evaluate clustering quality using homogeneity, completeness, and v-measure scores.

    Args:
        true_labels (np.ndarray): Ground truth labels of shape (N,).
        predicted_labels (np.ndarray): Predicted cluster labels of shape (N,).
        ignore_noise (bool): If True, ignore samples where predicted_labels == -1.

    Returns:
        Dict[str, float]: Dictionary with keys:
            - "homogeneity"
            - "completeness"
            - "v_measure"

    Notes:
        - Noise points (predicted_labels == -1) are excluded if ignore_noise=True.
        - Prints results in a readable format.
    """
    # Input validation
    if not isinstance(true_labels, np.ndarray):
        raise TypeError(f"true_labels must be np.ndarray, got {type(true_labels)}")
    if not isinstance(predicted_labels, np.ndarray):
        raise TypeError(f"predicted_labels must be np.ndarray, got {type(predicted_labels)}")
    if true_labels.shape[0] != predicted_labels.shape[0]:
        raise ValueError("true_labels and predicted_labels must have the same length")

    # Handle noise points
    if ignore_noise:
        mask = predicted_labels != -1
        true_labels = true_labels[mask]
        predicted_labels = predicted_labels[mask]

    # Global metrics
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    completeness = completeness_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)

    results = {
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
    }

    # Print results
    print("Clustering Quality Evaluation:")
    print(f"  Homogeneity: {homogeneity:.4f}")
    print(f"  Completeness: {completeness:.4f}")
    print(f"  V-measure: {v_measure:.4f}")
    # Per-cluster homogeneity 
    cluster_scores = {} 
    a = []
    for cluster_id in np.unique(predicted_labels): 
        mask = predicted_labels == cluster_id 
        labels = true_labels[mask]
        counts = Counter(labels)
        # h = homogeneity_score(true_labels[mask], predicted_labels[mask]) 
        if len(labels) > 0: 
            majority_class, majority_count = counts.most_common(1)[0] 
            h = majority_count / len(labels) 
        else: 
            majority_class, h = None, 0.0
        print(f" class_{cluster_id}: completeness={round(h, 4)}, majority_cluster={majority_class}")
        cluster_scores[f"cluster_{cluster_id}"] = {"homegeneity": round(h, 4), "majority_cluster": majority_class if majority_class is not None else None }
        a.append( {"cluster id":cluster_id, "homegeneity": round(h, 4), "majority_cluster": majority_class if majority_class is not None else None })
    df_string = pd.json_normalize(a)
    df_string.to_csv("results/cluster_scores.csv")

    # Per-class completeness 
    class_scores = {} 
    a = []
    for class_id in np.unique(true_labels): 
        mask = true_labels == class_id 
        # c = completeness_score(true_labels[mask], predicted_labels[mask]) 
        # class_scores[f"class_{class_id}"] = round(c, 4) 
        clusters = predicted_labels[mask]
        counts = Counter(clusters) 
        if len(clusters) > 0: 
            majority_cluster, majority_count = counts.most_common(1)[0] 
            c = majority_count / len(clusters) 
        else: 
            majority_cluster, c = None, 0.0
        print(f"class_{class_id} completeness: {round(c, 4)}, majority_cluster: {majority_cluster}" )
        class_scores[f"class_{class_id}"] = { "completeness": round(c, 4), "majority_cluster": int(majority_cluster) if majority_cluster is not None else None }
        a.append({"class_id":class_id, "completeness": round(c, 4), "majority_cluster": int(majority_cluster) if majority_cluster is not None else None })
    df_string = pd.json_normalize(a)
    df_string.to_csv("results/class_scores.csv")
    # Combine all metrics 
    results1 = { 
        "homogeneity": round(homogeneity, 4), 
        "completeness": round(completeness, 4), 
        "v_measure": round(v_measure, 4), 
        "per_cluster_homogeneity": cluster_scores, 
        "per_class_completeness": class_scores } 
    
    # Save to file if requested 
    if save_path: 
        try: 
            with open(save_path, "w") as f: 
                json.dump(results1, f, indent=2) 
            print(f"Saved metrics to {save_path}") 
        except Exception as e:
            print(f"[Error] Failed to save metrics: {e}")

    return results
# Clustering Quality Evaluation:
#   Homogeneity: 1.0000
#   Completeness: 1.0000
#   V-measure: 1.0000
# {'homogeneity': 1.0, 'completeness': 1.0, 'v_measure': 1.0}
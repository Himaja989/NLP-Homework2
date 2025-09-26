# Q5 — Evaluation Metrics from a Multi-Class Confusion Matrix

import numpy as np

# Confusion matrix given in the question:
# Rows = Predicted, Columns = Gold
#          Gold:  Cat   Dog   Rabbit
conf_matrix = np.array([
    [5, 10, 5],   # Predicted Cat
    [15, 20, 10], # Predicted Dog
    [0, 15, 10]   # Predicted Rabbit
])

# Class labels
classes = ["Cat", "Dog", "Rabbit"]

def compute_metrics(conf_matrix, classes):
    num_classes = len(classes)
    precision = {}
    recall = {}

    # Per-class precision and recall
    for i, cls in enumerate(classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[i, :].sum() - TP
        FN = conf_matrix[:, i].sum() - TP

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        precision[cls] = prec
        recall[cls] = rec

    # Macro average
    macro_precision = np.mean(list(precision.values()))
    macro_recall = np.mean(list(recall.values()))

    # Micro average
    TP_sum = np.trace(conf_matrix)
    FP_sum = conf_matrix.sum(axis=1).sum() - TP_sum
    FN_sum = conf_matrix.sum(axis=0).sum() - TP_sum

    micro_precision = TP_sum / (TP_sum + FP_sum)
    micro_recall = TP_sum / (TP_sum + FN_sum)

    return precision, recall, macro_precision, macro_recall, micro_precision, micro_recall

# Run metrics computation
precision, recall, macro_p, macro_r, micro_p, micro_r = compute_metrics(conf_matrix, classes)

# Print results
print("Per-class Precision:")
for cls in classes:
    print(f"  {cls}: {precision[cls]:.4f}")

print("\nPer-class Recall:")
for cls in classes:
    print(f"  {cls}: {recall[cls]:.4f}")

print("\nMacro-averaged Precision:", round(macro_p, 4))
print("Macro-averaged Recall:", round(macro_r, 4))

print("\nMicro-averaged Precision:", round(micro_p, 4))
print("Micro-averaged Recall:", round(micro_r, 4))

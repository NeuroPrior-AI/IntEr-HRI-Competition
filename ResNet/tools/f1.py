def calculate_f1_score(confusion_matrix):
    # Extract true positive, false positive, true negative, false negative
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]

    # Calculate precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

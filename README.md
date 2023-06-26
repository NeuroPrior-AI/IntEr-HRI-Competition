# IntEr-HRI-Competition

# Data Processing
1. Filter: 0.1 - 50 Hz bandpass filter

# Model Training
1. Ensemble Model
- An ensemble of machine learning models comprising MLP, logistic regression, SVC, random forest, k-NN, XGBoost, LSTM, CNN, and a grid-search-optimized XGBoost are bundled in a pipeline with feature extraction and transformation steps.

# 10 Fold Validation
1. Accuracy.txt
- The Accuracy.txt file contains the classification accuracy for each validation cycle.

2. Confusion Matrix
- The Confusion Matrix contains the result of the Ensemble model. Each confusion matrix is normalized.
- The Predicted label presents the predict output from the label, True label shows the actual label in validation dataset.
- "S 96" shows the event for Error Induction, "S 80" presents the event for Button press, "no error" shows the event for other labels.

3. F1 Score
- f1_score.txt shows the result of F1 score. 

# Time Point Predict Algorithum
1. Algorithum


# IntEr-HRI-Competition

# Dataset
Create folder `Dataset` under root, and put data inside.

# Data Preprocessing
```
cd Preprocess
```
and then run
```
python process.py
```

# Model Training
1. Ensemble Model
- An ensemble of machine learning models comprising MLP, logistic regression, SVC, random forest, k-NN, XGBoost, and a grid-search-optimized XGBoost are bundled in a pipeline with feature extraction and transformation steps.

```
cd Models/ensemble_model
```
and then run
```
python ensemble.py
```

2. Custom ResNet Model
- The ResNet model starts with a 1-D convolutional layer, followed by batch normalization and a ReLU activation. A custom ResidualBlock structure is defined, which consists of two convolutional layers with batch normalization and ReLU activation, as well as a skip connection path from the input to the output. The residual block is applied twice in sequence. After passing through the residual blocks, the output is passed to an adaptive average pooling layer, flattened, and finally fed to a fully connected linear layer for binary classification.

# 10 Fold Validation
1. Accuracy.txt
- The Accuracy.txt file contains the classification accuracy for each validation cycle.

2. Confusion Matrix
- The Confusion Matrix contains the result of the Ensemble model. Each confusion matrix is normalized.
- The Predicted label presents the predict output from the label, True label shows the actual label in validation dataset.
- "S 96" shows the event for Error Induction, "S 80" presents the event for Button press, "no error" shows the event for other labels.

3. F1 Score
- f1_score.txt shows the result of F1 score. 

# Time Point Predict algorithm

run
```
python Algorithms/probmap.py
```

# Testing
1. Offline test
```
python Testing/offline_test.py 
```

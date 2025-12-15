EEG/BCI Classification: NCC vs LDA (with 10-fold Cross-Validation)

This repository contains a small Python experiment for classifying BCI/EEG data using two linear classifiers:

Nearest Centroid Classifier (NCC) and Linear Discriminant Analysis (LDA)

The script:
1. loads a MATLAB dataset (bcidata.mat),
2. trains NCC and LDA on a random train/test split and saves a histogram comparison plot,
3. runs 10-fold cross-validation (default with LDA),
4. saves a boxplot of train/test CV accuracies.

Files:
lda_proect.py: implementation and experiments
bcidata.mat: dataset file (must be present locally to run)

Output files:
 - ncc-lda-comparison.pdf — histogram of classifier scores on the test set and accuracies
 - figure2_boxplot_accy.pdf — boxplot of train vs. test accuracy across CV folds

Dataset format (bcidata.mat)
The code expects the .mat file to contain:
X: EEG data array (the script reshapes it into (dims × samples))
Y: labels (converted to {-1, +1})


Requirements:
Python 3.8+ recommended
NumPy
SciPy
Matplotlib

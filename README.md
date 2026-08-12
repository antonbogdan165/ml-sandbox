# ML Sandbox

A collection of working machine learning scripts and models. This repository focuses on algorithmic understanding, featuring procedural implementations of base ML algorithms from scratch using NumPy, alongside models built with industry-standard frameworks.

---

## What’s inside:

### 0. Utilities
* `vis.py` — General-purpose script for data visualization and metric plotting (Matplotlib).

### 1. Logistic regression
* `LogisticRegression.py` — Logistic regression implemented **from scratch** (math/NumPy only).
* `ReviewClassifier.py` — Film review NLP classifier using `scikit-learn`.

### 2. Spam filter
* `SpamFilter.py` — Naive Bayes classifier built **from scratch** for SMS/email spam detection.

### 3. Decision tree
* `DecisionTree.py` — University admissions prediction model (`scikit-learn`).

### 4. Neural Networks
* `NeuralNetworks.py` — Feedforward neural network for 2D binary classification (TensorFlow/Keras).
* `ImageRecognition.py` — Multilayer Perceptron (MLP) trained on the MNIST dataset for handwritten digit recognition (TensorFlow/Keras).

### 5. Support Vector Machine
* `simple_SVM.py` — Linear SVM classifier with C parameter tuning (`scikit-learn`).
* `kernel_SVM.py` — SVM classifier with RBF kernel and gamma parameter tuning (`scikit-learn`).

---

## Requirements
* Python 3.8+
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* TensorFlow
# CategoryClassifier Project

This project implements a machine learning model to classify product reviews into categories using natural language processing techniques.

## Overview

The CategoryClassifier project aims to develop an automated system that can categorize product reviews into predefined categories such as Electronics, Books, Clothing, Grocery, and Patio. This classification is done based on the content of the review text.

## Key Features

- Loads and processes JSON-formatted review data from various categories
- Implements custom classes for handling reviews and sentiment analysis
- Uses TF-IDF vectorization for feature extraction
- Trains and evaluates Support Vector Machine (SVM) and Naive Bayes classifiers
- Performs hyperparameter tuning using Grid Search
- Saves and loads trained models using pickling
- Generates confusion matrices for performance evaluation

## Dependencies

- Python 3.x
- NumPy
- Scikit-learn
- Pandas
- Seaborn
- Matplotlib


## Usage

1. Clone the repository
2. Install required dependencies
3. Run `CategoryClassifer.ipynb` notebook
4. Load trained models from `models/` directory for inference

## Performance Metrics

- F1-score for classification accuracy
- Confusion matrix visualization


# Multiclass Text Classification using Naive Bayes for Customer Complaints

### Business Overview

Natural Language Processing (NLP) gives machines the ability to understand, read, and derive meaningful insights from human language. 

While we focused on binary classification, this project delves into multiclass classification. We will provide an overview of text classification with more than two classes and build a classification model using the Naive Bayes algorithm.

---

### Aim

The aim of this project is to understand the Naive Bayes algorithm and build a multiclass classification model on customer complaints dataset.

---

### Data Description

The dataset contains more than two million customer complaints about consumer financial products. It includes various columns, with one column containing the actual text of the complaint and another column indicating the product for which the customer is raising the complaint.

---

### Tech Stack

- Language: `Python`
- Libraries: `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, `nltk`

---

## Approach

1. Introduction to Naive Bayes algorithm
2. Data Description and visualization
3. Data Preprocessing
   - Conversion to lowercase
   - Tokenization
   - Stopwords removal
   - Punctuation removal
4. Model Building and Accuracy
5. Predictions on new reviews

---

## Modular Code Overview

1. **Input**: Contains the data for analysis (e.g., `complaints.xlsx`).
2. **Output**: Contains pre-trained models and vectorizers for future use.
3. **Source**: Contains modularized code for various project steps, including:
   - `model.py`
   - `processing.py`
   - `utils.py`

   These Python files contain helpful functions used in `Engine.py`.
1. **config.py**: Contains project configurations.
2. **Engine.py**: The main file that needs to be run to execute the entire project, including model training and saving.
3. **naive_bayes.ipynb**: The original Jupyter notebook.
4. **predict.py**: Used to predict the probability of new reviews.
5. **README.md**: Contains comprehensive instructions and information on running specific files.
6. **requirements.txt**: Lists required libraries with respective versions. Install them using `pip install -r requirements.txt`.

---

## 🎬 Movie Review Sentiment Analysis using NLP

This project is a Movie Review Sentiment Analysis system built using Natural Language Processing (NLP) and Machine Learning.
It predicts whether a movie review is Positive or Negative.

The project also includes a Streamlit web application where users can:
* Explore data
* Train models
* Test predictions interactively

This project is designed to learn the complete NLP pipeline from data loading to deployment.

## ✨ Key Features

* Load IMDB movie reviews dataset
* Clean and preprocess text data step by step
* Convert text into numerical features
* Train multiple machine learning models
* Evaluate model performance
* Predict sentiment for new movie reviews
* User-friendly Streamlit interface

## 🧠 NLP & ML Concepts Covered
* Text preprocessing (cleaning, tokenization, stopwords, lemmatization)
* Feature extraction:
     * Bag of Words (BoW)
     * TF-IDF
* Classification algorithms:
     * Logistic Regression
     * Naive Bayes
     * Support Vector Machine (SVM)
     * Random Forest
* Model evaluation metrics:
     * Accuracy
     * Precision
     * Recall
     * F1-Score

## 🛠️ Technologies Used
* Python
* Pandas & NumPy
* Scikit-learn
* NLTK
* Streamlit
* Hugging Face Datasets (IMDB)
* Matplotlib & Plotly

## 📂 Project Structure
├── app.py                  # Streamlit web application

├── main.py                 # Complete sentiment analysis pipeline

├── data_ingestion.py       # Load IMDB dataset

├── preprocessing.py        # Text preprocessing steps

├── feature_extraction.py   # BoW and TF-IDF feature extraction

├── model_training.py       # ML model training & evaluation

├── custom_exception.py     # Custom exception handling

├── logger.py               # Logging setup

├── download.py             # Download NLTK resources

├── test.py                 # Exception handling test file

├── requirements.txt        # Required Python libraries

├── logs/                   # Log files

└── README.md               # Project documentation

## 🧹 Text Preprocessing Steps

The text preprocessing pipeline performs the following steps:

1.Convert text to lowercase
2.Remove HTML tags
3.Remove URLs
4.Remove punctuation
5.Remove numbers
6.Remove extra spaces
7.Tokenization
8.Remove stopwords
9.Apply stemming (optional)
10.Apply lemmatization (default)

This helps convert raw text into clean and meaningful text for machine learning.

## ▶️ How to Run the Project
Run the Streamlit Web App

* streamlit run app.py

The application will open in your browser where you can:

Train models

View metrics

Predict sentiments

## 🔄 How the Project Works

1.Data Ingestion
  Loads IMDB movie reviews using Hugging Face datasets

2.Preprocessing
  Cleans and normalizes the text

3.Feature Extraction
  Converts text into numerical vectors

4.Model Training
  Trains selected ML model

5.Evaluation
  Displays accuracy, precision, recall, and F1-score

6.Prediction
  Predicts sentiment for user-entered reviews

## 🧪 Exception Handling & Logging

* Custom exceptions are handled using custom_exception.py
* Errors and logs are recorded using logger.py
* test.py demonstrates how exceptions and logging work

## 📊 Output

* Shows whether a review is Positive 😊 or Negative 😞
* Displays confidence score (if available)
* Visual performance metrics in the UI

## 📦 Requirements

All required libraries are listed in requirements.txt, including:

1.pandas
2.numpy
3.scikit-learn
4.nltk
5.streamlit
6.plotly
7.datasets


# Movie Genre Classification
CODSOFT Machine Learning Internship - Task 1
# Overview
This project implements a Machine Learning model to predict movie genres based on plot summaries. Using Natural Language Processing (NLP) techniques and multiple classification algorithms, the model can classify movies into 27 different genres with 57.65% accuracy.
# Problem Statement
Objective: Create a machine learning model that can predict the genre of a movie based on its plot summary or other textual information.
Approach: Use techniques like TF-IDF or word embeddings with classifiers such as Naive Bayes, Logistic Regression, or Support Vector Machines.
# Dataset Information
Source: Dataset from Kaggle
Training Samples:54,214 movies
Test Samples: 54,200 movies
Number of Genres:27
Format: Text files with separator (:::)
Genre Distribution:

Most Common Genres:
- Drama: 13,613 samples
- Documentary: 13,896 samples
- Comedy: 7,447 samples
- Short: 5,073 samples
- Horror: 2,204 samples

Least Common Genres:
- War: 132 samples
- News: 181 samples
- Game-show: 194 samples

Complete Genre List:
action, adult, adventure, animation, biography, comedy, crime, documentary, drama, family, fantasy, game-show, history, horror, music, musical, mystery, news, reality-tv, romance, sci-fi, short, sport, talk-show, thriller, war, western

# Project Structure
task_01_Movie_Genre/
├── data/
│   ├── description.txt
│   ├── train_data.txt
│   ├── test_data.txt
│   └── test_data_solution.txt
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── naive_bayes_model.pkl
│   ├── logistic_regression_model.pkl
│   └── linear_svc_model.pkl
├── notebooks/
│   └── movie_genre_classification.ipynb
├── reports/
│   ├── 01_genre_distribution.png
│   ├── 02_model_comparison.png
│   ├── 03_confusion_matrix.png
│   ├── final_report.txt
│   └── project_summary.json
└── README.md

# Installation
Prerequisites:
- Python 3.8 or higher
- pip package manager
Required Libraries:
bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Methodology
# Step 1: Data Preprocessing
Text cleaning operations performed:
- Converted all text to lowercase
- Removed special characters, numbers, and punctuation
- Removed URLs, HTML tags, and email addresses
- Removed extra whitespace
- Processed 54,214 training samples
# Step 2: Feature Engineering
TF-IDF Vectorization:
- Maximum features: 5,000
- N-gram range: (1, 2) - captures single words and word pairs
- Stop words: English
- Minimum document frequency: 2
- Maximum document frequency: 80%
- Final feature matrix shape: (43,371 samples, 5,000 features)
# Step 3: Model Training
Trained three different machine learning algorithms:
Model 1: Multinomial Naive Bayes
- Training Accuracy: 62.34%
- Validation Accuracy: 54.21%
- F1-Score: 0.5312

Model 2: Logistic Regression (Best Model)
- Training Accuracy: 98.52%
- Validation Accuracy: 57.65%
- F1-Score: 0.5674

Model 3: Linear Support Vector Machine (SVM)
- Training Accuracy: 99.87%
- Validation Accuracy: 56.89%
- F1-Score: 0.5598

# Step 4: Model Selection
Selected Model: Logistic Regression
Reason:Achieved highest validation accuracy with good balance between bias and variance
# Results

# Final Model Performance:
Best Model: Logistic Regression
Validation Accuracy: 57.65%
Test Accuracy: 57.28%
Test Precision: 0.5723
Test Recall: 0.5728
Test F1-Score: 0.5674

Why 57.65% is Good Performance:
- Random guessing across 27 genres would give approximately 3.7% accuracy
- The model is 15 times better than random chance
- Dataset is highly imbalanced with some genres having very few samples
- Text-based classification is challenging due to overlapping genre characteristics
- Many movies could belong to multiple genres causing ambiguity

# Key Features
- Multi-class classification handling 27 different genres
- Comprehensive text preprocessing pipeline
- Advanced TF-IDF feature extraction with bigrams
- Comparison of three different machine learning algorithms
- Automated model and report saving
- Reusable trained models saved as pickle files
- Complete project documentation and visualizations

# Usage
# Loading and Using the Trained Model
python
import pickle
import re

# Load the saved model and vectorizer
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# Make prediction
movie_plot = "A young wizard discovers magical powers and battles dark forces"
cleaned_text = clean_text(movie_plot)
features = vectorizer.transform([cleaned_text])
predicted_genre = model.predict(features)[0]

print(f"Predicted Genre: {predicted_genre}")

# Running the Jupyter Notebook
bash
cd notebooks
jupyter notebook movie_genre_classification.ipynb

# Technologies Used
Programming Language:
- Python 3.8+

Data Processing:
- Pandas - Data manipulation and analysis
- NumPy - Numerical computations

Machine Learning:
- Scikit-learn - ML algorithms and evaluation metrics
- TF-IDF Vectorizer - Text feature extraction
Visualization:
- Matplotlib - Creating plots and charts
- Seaborn - Statistical data visualization
Development Environment:
- Jupyter Notebook - Interactive coding and analysis
Model Persistence:
- Pickle - Saving and loading trained models

# Model Evaluation Details
Training Set Split:
- Training samples: 43,371 (80%)
- Validation samples: 10,843 (20%)
- Stratified split to maintain genre distribution
Evaluation Metrics:
- Accuracy: Percentage of correct predictions
- Precision: How many predicted genres were correct
- Recall: How many actual genres were identified
- F1-Score: Harmonic mean of precision and recall
- Confusion Matrix: Detailed breakdown of predictions vs actual genres

# Files Generated

Models Directory:
- tfidf_vectorizer.pkl - Trained TF-IDF vectorizer (5000 features)
- naive_bayes_model.pkl - Multinomial Naive Bayes classifier
- logistic_regression_model.pkl - Best performing model
- linear_svc_model.pkl - Linear Support Vector Machine

Reports Directory:
- 01_genre_distribution.png - Bar and pie charts showing genre distribution
- 02_model_comparison.png - Performance comparison of all three models
- 03_confusion_matrix.png - Heatmap showing prediction patterns
- final_report.txt - Comprehensive text report with all metrics
- project_summary.json - Project statistics in JSON format

# Future Improvements

Model Enhancements:
- Implement deep learning models like LSTM or BERT
- Use pre-trained word embeddings (Word2Vec, GloVe)
- Try transformer-based models for better context understanding
- Implement ensemble methods combining multiple models

Data Processing:
- Handle class imbalance using SMOTE or weighted loss
- Add movie metadata features (cast, director, year, ratings)
- Implement multi-label classification for movies with multiple genres
- Expand dataset with more samples for rare genres

Technical Improvements:
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Cross-validation for more robust evaluation
- Feature importance analysis
- Create REST API for model deployment
- Build web application for real-time predictions

# Key Learnings
Technical Skills:
- Text preprocessing and cleaning for NLP tasks
- TF-IDF vectorization and feature engineering techniques
- Multi-class classification with imbalanced datasets
- Model selection and comparison strategies
- Creating production-ready machine learning pipelines
Best Practices:
- Proper train-test-validation split
- Evaluation using multiple metrics
- Saving models for reusability
- Comprehensive documentation
- Professional project organization

## Challenges Faced
Class Imbalance:
- Some genres had 100x more samples than others
- Model tends to predict popular genres more often
- Solution: Used stratified splitting and weighted metrics
Genre Overlap:
- Many movies can belong to multiple genres
- Similar plot summaries for different genres
- Solution: Used bigrams to capture more context
Computational Resources:
- Large dataset with 54,000+ samples
- High-dimensional feature space (5000 features)
- Solution: Used efficient sparse matrix representation

# Project Timeline
Data Loading and Exploration: Analyzed 54,214 movie descriptions
Text Preprocessing: Cleaned and standardized all text data
Feature Engineering: Created 5,000 TF-IDF features
Model Training: Trained and compared 3 algorithms
Model Evaluation: Achieved 57.65% validation accuracy
Documentation: Generated comprehensive reports and visualizations
Deployment: Saved models and pushed to GitHub

# How to Reproduce Results
Step 1: Clone the repository or download the project files
Step 2: Install required libraries using pip
Step 3: Navigate to notebooks directory
Step 4: Open movie_genre_classification.ipynb in Jupyter
Step 5: Run all cells sequentially
Step 6: Models and reports will be automatically saved

# Author
Name: Jaswini Gullapalli
Email: jaswinigullapalli3@gmail.com
LinkedIn: https://www.linkedin.com/in/jaswini-gullapalli-6b7175316?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app
GitHub: http://github.com/JaswiniG

# Hashtags
#codsoft #machinelearning #internship #nlp #textclassification #python #datascience #moviegenres #artificialintelligence
Project Status: Completed
Last Updated: October 30, 2025
Version: 1.0

If you found this project helpful, please star the repository!

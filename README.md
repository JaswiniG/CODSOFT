Spam SMS Detection
This project is part of my CodSoft Machine Learning Internship. The goal is to build an AI model that classifies SMS messages as either spam or legitimate (ham) using text processing and machine learning techniques.

Project Overview:
The model uses TF-IDF vectorization to convert text into numerical features and applies multiple classifiers such as Naive Bayes, Logistic Regression, and Support Vector Machines to detect spam messages.

Techniques Used:
Natural Language Processing (NLP)
Text preprocessing (stopword removal, tokenization, cleaning)
TF-IDF vectorization
Machine Learning Classification
Tech Stack
Python
Pandas, NumPy
Scikit-learn
NLTK
Matplotlib, Seaborn

Project Structure:
CodSoft/
│
├── Task_04_Spam_SMS/
│   ├── data/             # Dataset (spam.csv)
│   ├── models/           # Saved model (spam_model.pkl)
│   ├── notebooks/        # Jupyter Notebook files
│   ├── reports/          # Any reports or visualizations
│   └── src/              # Python scriptsHow to Run

How to run?
1. Clone the repository
git clone https://github.com/JaswiniG/CODSOFT.git
2. Navigate to the project folder
cd CODSOFT/Task_04_Spam_SMS
3. Install dependencies
pip install -r requirements.txt
4. Run the notebook or script to train/test the model.

Output:
The model predicts whether an SMS message is spam or ham, and achieves high accuracy using TF-IDF features.

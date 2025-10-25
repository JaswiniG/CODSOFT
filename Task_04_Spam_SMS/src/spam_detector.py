import joblib

# Load the trained model and vectorizer
model = joblib.load("../models/spam_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

# Function to predict whether a message is spam or not
def predict_message(message):
    data = vectorizer.transform([message])
    prediction = model.predict(data)[0]
    return "Spam" if prediction == 1 else "Not Spam"

if _name_ == "_main_":
    samples = [
        "Congratulations! You have won a $500 gift card. Click here to claim your prize!",
        "Hey, are we still meeting for lunch today?",
        "You have been selected for a free vacation. Reply YES to claim."
    ]

    for text in samples:
        print(f"Message: {text}")
        print(f"Prediction: {predict_message(text)}\n")

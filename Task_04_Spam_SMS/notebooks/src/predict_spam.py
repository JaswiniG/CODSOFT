
import joblib

model = joblib.load('../models/spam_model.pkl')
vectorizer = joblib.load('../models/vectorizer.pkl')

def predict_sms(text):
    processed_text = vectorizer.transform([text])
    pred = model.predict(processed_text)
    return 'Spam' if pred[0] else 'Ham'

if _name_ == '_main_':
    sample_sms = 'Congratulations! You won a free ticket.'
    print('Sample SMS prediction:', predict_sms(sample_sms))

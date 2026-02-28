import os
import string
import joblib
import cv2
import pytesseract
from nltk.stem import SnowballStemmer
# 1. Setup & Preprocessing Engine
# Must match training exactly so the model recognizes the vocabulary
stemmer = SnowballStemmer("english")
def clean(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return " ".join([stemmer.stem(w) for w in text.split()])
# Will check the text inside an image and extract it
#I don't know much about this image part as I vibe coded this part, sorry :(
def extract_image_text(path):
    if not os.path.exists(path):
        return ""
    # Read image and convert to grayscale to remove color noise, basically removing wild pixals that will make it difficult for the computer to read the test in an image.
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Extract the hidden text, they can't hide from us!!!!
    extracted = pytesseract.image_to_string(thresh)
    return extracted if extracted.strip() else ""
# Load the Trained Brain
print("Initializing Spam & Phishing Detector...")
try:
    ensemble = joblib.load('model.pkl')
    vec = joblib.load('vec.pkl')
    sel = joblib.load('sel.pkl')
except Exception as e:
    print(f"Error: Missing .pkl files. Please place them in this folder.\nDetails: {e}")
    exit()
while True:
    text = input("\nEmail Text (press Enter to skip, type 'exit' to quit): ")
    if text.lower() == 'exit':
        print("Ok") #Couldn't think what to print here :)
        break
    img_path = input("Image Path (press Enter to skip): ")
    # Combine text from both the email body and the image attachment
    vision_text = extract_image_text(img_path) if img_path else ""
    full_content = text + " " + vision_text
    if not full_content.strip():
        print("No input detected. Please try again.")
        continue
    cleaned = clean(full_content)
    vectorized = vec.transform([cleaned])
    features = sel.transform(vectorized)
    # 6. Prediction & Probability
    pred = ensemble.predict(features)[0]
    probs = ensemble.predict_proba(features)[0]
    spam_conf = probs[1] * 100
    safe_conf = probs[0] * 100
    if pred == 1:
        print(f"\nSPAM/PHISHING \nProbability: {spam_conf:.2f}%")
    else:
        print(f"\nSAFE \nProbability: {safe_conf:.2f}%")

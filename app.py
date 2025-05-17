from flask import Flask, json, render_template, request, jsonify
import joblib
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os
load_dotenv()

app = Flask(__name__)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def generate_response(user_query):
    custom_template = """
    you are cyberbullying detector if you detect cyberbullying return string 'Cyberbullying detected!'.
    detect cyberbullying even for minute harrasment. 
    Sentence: {question}
    Answer:
    """
    filled_prompt = custom_template.format(question=user_query)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(filled_prompt)
    return response.text
# Load the saved CountVectorizer and model
count_vector = joblib.load('vectorizer.pkl')  # You must have saved this during training
model_filename = 'cb_sgd_final.sav'
loaded_model = joblib.load(model_filename)

def detect_cyberbullying(text):
    # Transform the input text using the loaded vectorizer
    preprocessed_text = count_vector.transform([text])
    prediction = loaded_model.predict(preprocessed_text)

    # Return result based on prediction
    if prediction[0] == 1:
        return "Cyberbullying detected!"
    else:
        response = generate_response(text)
        # print(response)
        if response.strip() == "Cyberbullying detected!":
            # print("Cyberbullying detected!!")
            return "Cyberbullying detected!!"
        else:
            # print("No cyberbullying detected.")
            return "No cyberbullying detected."

@app.route("/",methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def detect():
    data = request.json
    user_text = data.get("text","")

    if not user_text:
        return jsonify({"error": "No text provided"}), 400
    result = detect_cyberbullying(user_text)
    print(result)
    return jsonify({"result":result}), 201



if __name__== "__main__":
    app.run(debug=True)
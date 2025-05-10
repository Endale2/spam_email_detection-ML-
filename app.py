
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)
model = joblib.load("spam_classifier.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    input_text = ""
    if request.method == "POST":
        input_text = request.form["email_text"]
        result = model.predict([input_text])[0]
        prediction = "Spam" if result == 1 else "Not Spam"
    return render_template("index.html", prediction=prediction, email_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)

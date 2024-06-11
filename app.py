from flask import Flask, request, jsonify, render_template
import joblib
from pyforest import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

tokenizer = AutoTokenizer.from_pretrained("ozayezerceli/t5-base-title-generator")
# Create a Flask app
app = Flask(__name__)


# Load the trained ML model
model= joblib.load(open("model.pkl", "rb"))

# Define a route for generating titles
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the request
    input_text = request.form['input-text']
    num_titles = int(request.form['num-titles'])
    temperature = float(request.form['temperature'])
    inputs = ["summarize: " + input_text]
    generated_titles = generate_titles(inputs, num_titles, temperature)
    for idx, title in enumerate(generated_titles, 1):
        print(f"Title {idx}: {title}")

    # Return the generated title as a response
    return render_template("results.html", titles=generated_titles)

def generate_titles(input_text, num_titles, temperature):
    inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt")
    titles = []
    for _ in range(num_titles):
        output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64, temperature=temperature)
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
        titles.append(predicted_title)
    return titles

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

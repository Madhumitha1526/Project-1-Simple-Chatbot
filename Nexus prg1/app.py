from flask import Flask, request, jsonify, render_template
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Store context
context_memory = {}

# Basic chatbot responses
responses = {
    "greet": "Hello! How can I help you today?",
    "questions": {
        "how are you": "I'm a bot, so I don't have feelings, but thanks for asking!",
        "what is your name": "I am a simple chatbot created for demonstration purposes.",
        "what can you do": "I can chat with you and remember our conversations.",
        "tell me a joke": "Why did the scarecrow become a successful neurosurgeon? Because he was outstanding in his field!",
        "wish me a luck":"Good luck you have a wonderfull day",
        "goodbye": "Goodbye! Have a great day!"
    },
    "fallback": "I'm sorry, I didn't understand that. Can you please rephrase?"
}

greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]

def preprocess_text(text):
    # Tokenize, remove stop words and punctuation, and lemmatize
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return render_template('index.html')

    user_id = request.json.get('user_id')
    message = request.json.get('message').lower()

    print(f"Received message: {message}")

    if user_id not in context_memory:
        context_memory[user_id] = {"greeted": False, "messages": []}

    context = context_memory[user_id]

    if not context["greeted"]:
        reply = responses["greet"]
        context["greeted"] = True
    else:
        tokens = preprocess_text(message)
        print(f"Processed tokens: {tokens}")

        # Check for greeting
        if any(greet in tokens for greet in greetings):
            reply = responses['greet']
        else:
            # Match the processed tokens to the closest response key
            matched_response = None
            for question_key, response in responses['questions'].items():
                question_tokens = preprocess_text(question_key)
                if tokens == question_tokens:
                    matched_response = response
                    break

            if matched_response:
                reply = matched_response
            else:
                reply = responses['fallback']

    context["messages"].append(message)
    context_memory[user_id] = context

    return jsonify({"response": reply, "context": context["messages"]})

if __name__ == '__main__':
    app.run(debug=True)

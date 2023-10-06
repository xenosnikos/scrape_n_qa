from flask import Flask, render_template, request, jsonify
from app import create_embeddings, get_answer  ,socketio # Import the functions from app.py

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_embeddings', methods=['POST'])
def create():
    sitemap_url = request.form['sitemap_url']
    create_embeddings(sitemap_url)  # Call the function to create embeddings
    return jsonify(status="success")  # Return a success response

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    chat_history = []  # You can manage chat history as per your requirements
    answer = get_answer(question, chat_history)  # Call the function to get answer
    return jsonify(answer=answer)  # Return the answer as JSON

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
    app.run(debug=True)

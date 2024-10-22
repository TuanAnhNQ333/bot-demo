from chatterbot import Chatbot
from chatterbot.trainers import ChatterBotCorpusTrainer
Chatbot = Chatbot('MyChatbot')
trainer = ChatterBotCorpusTrainer(Chatbot)
trainer.train("chatterbot.corpus.english")
response = Chatbot.get_response("Hello World!")
print(response)
from chatterbot.trainers import ListTrainer
trainer = ListTrainer(Chatbot)
trainer.train([
    "How are you?",
    "I am good.",
    "That is good to hear.",
    "Thank you",
    "You're welcome."
])
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(Chatbot.get_response(userText))

if __name__ == "__main__":
    app.run()
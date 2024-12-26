# app.py

from flask import Flask, render_template, request
from chatbot_model import create_chatbot
from api_integration import get_weather

app = Flask(__name__)
chatbot = create_chatbot()

@app.route("/") 
def home():
    return render_template("index.html")

@app.route("/get") 
def get_bot_response():
    userText = request.args.get('msg')
    if "weather" in userText.lower():
        city = userText.split("weather in")[-1].strip()
        response = get_weather(city)
    else:
        response = chatbot.get_response(userText)
    return str(response)

if __name__ == "__main__":
    app.run(debug=True)

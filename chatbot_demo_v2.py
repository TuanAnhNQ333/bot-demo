from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
from flask import Flask, render_template, request

# Khởi tạo chatbot
chatbot = ChatBot('MyChatbot')

# Huấn luyện với dữ liệu chuẩn
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")

# Huấn luyện với dữ liệu tùy chỉnh
trainer = ListTrainer(chatbot)
trainer.train([
    "How are you?",
    "I am good.",
    "That is good to hear.",
    "Thank you.",
    "You're welcome."
])

# Khởi tạo Flask
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    response = chatbot.get_response(userText)
    return str(response)

if __name__ == "__main__":
    app.run(debug=True)  # Bật chế độ debug để dễ dàng phát hiện lỗi

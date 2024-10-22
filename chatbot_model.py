# chatbot_model.py

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer

def create_chatbot():
    # Khởi tạo chatbot
    chatbot = ChatBot('MyChatbot')

    # Huấn luyện với dữ liệu chuẩn
    trainer = ChatterBotCorpusTrainer(chatbot)
    trainer.train("chatterbot.corpus.english")

    # Huấn luyện với dữ liệu tùy chỉnh
    trainer = ListTrainer(chatbot)
    custom_conversations = [
        "How are you?",
        "I am good.",
        "That is good to hear.",
        "Thank you.",
        "You're welcome.",
        "What is your name?",
        "I am a chatbot.",
        "Tell me a joke.",
        "Why did the scarecrow win an award? Because he was outstanding in his field!"
    ]
    trainer.train(custom_conversations)

    return chatbot

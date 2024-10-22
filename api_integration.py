# api_integration.py

import requests

def get_weather(city):
    api_key = 'ceefd2f6333be205b67bfcf63a5200b4'
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(base_url)
    data = response.json()

    if data['cod'] == 200:
        main = data['main']
        weather_desc = data['weather'][0]['description']
        temperature = main['temp']
        return f"Weather in {city}: {temperature}°C with {weather_desc}."
    else:
        return "City not found."

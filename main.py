import requests
from STT import speech_to_text
from XTTS import text_to_speech

headers = {"Content-Type": "application/json"}

response = requests.post(f"http://127.0.0.1:8000/generate?prompt=Who is Barack Obama?", headers=headers)
print(response.json()["response"])

response2 = requests.post(f"http://127.0.0.1:8000/generate?prompt=How tall is he?", headers=headers)
print(response2.json()["response"])




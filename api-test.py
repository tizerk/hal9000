import requests

url = "http://127.0.0.1:8000/generate?prompt=How are you doing?"
headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers)
print(response.json()["response"])

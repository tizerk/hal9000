import requests
from STT import STT
from TTS import TTS


headers = {"Content-Type": "application/json"}
server_url = "http://127.0.0.1:8000/generate?prompt="

if __name__ == "__main__":
    try:
        requests.get(f"http://127.0.0.1:8000/test", headers=headers)
    except Exception:
        print(
            "No response from Ollama, make sure both Ollama and the server are running."
        )

    tts_module = TTS()
    stt_module = STT()

    while True:
        user_input = stt_module.speech_to_text()
        print("User said:\n\t" + user_input)
        print("Asking HAL...")
        try:
            llm_response = requests.post(server_url + user_input, headers=headers)
            tts_module.text_to_speech(llm_response.json()["response"])
            print("HAL9000 said:\n\t" + llm_response.json()["response"])
        except requests.exceptions.JSONDecodeError:
            print(
                "No response from Ollama, make sure both Ollama and the server are running."
            )

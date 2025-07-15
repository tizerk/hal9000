import sys
import requests
import logging

logging.basicConfig(
    level=logging.WARN,
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
print("Starting HAL9000...")
from STT import STT
from TTS import TTS

headers = {"Content-Type": "application/json"}
server_url = "http://127.0.0.1:8000"

if __name__ == "__main__":

    tts_module = TTS()
    stt_module = STT()
    try:
        while True:
            user_input = stt_module.speech_to_text()
            print(f"User said:\n\t" + user_input + "\n")
            print("Asking HAL...\n")
            try:
                llm_response = requests.post(
                    f"{server_url}/generate?prompt={user_input}", headers=headers
                )
                tts_module.text_to_speech(llm_response.json()["response"])
                print("HAL9000 said:\n\t" + llm_response.json()["response"] + "\n")
            except requests.exceptions.JSONDecodeError:
                logger.error(
                    "No response from Ollama. Make sure both Ollama and the FastAPI server are running."
                )
                sys.exit(1)
    except KeyboardInterrupt:
        print("User interrupted, exiting...")
        sys.exit(0)

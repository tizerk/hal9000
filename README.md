# HAL 9000

![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)

This project brings HAL 9000 to life as a voice-driven AI assistant using a combination of real-time speech-to-text, a local large language model, MCP tooling, and finetuned text-to-speech inference.  HAL 9000 can run **100% offline.**

![HAL 9000 TUI Demo](https://github.com/YOUR_USERNAME/YOUR_REPO/blob/main/assets/hal9000_demo.gif)

## Table of Contents
- [HAL 9000](#hal-9000)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Environment Variables](#environment-variables)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)

## Requirements

This program runs several resource-intensive processes simultaneously. For a smooth experience, the following is recommended:
- **Python**: 3.12 (May work with newer versions, but I've only tested it with Python 3.12)
- **GPU**: An NVIDIA graphics card with at least 12GB of VRAM is highly recommended for faster performance. It is possible to run HAL with only a CPU, but it will be ***very*** slow.

## Installation

1.  **Clone the Repository:**
    ```
    git clone git@github.com:tizerk/hal9000.git
    ```
2. **Download the HAL 9000 TTS Model:**
   - Download the `hal9000.pth` file from [Hugging Face](https://huggingface.co/tizerk/hal9000/resolve/main/hal9000.pth?download=true)
   - Move the file to the following directory:
        `HAL9000/src/StyleTTS/Models`
        - There should be two files in this directory: `config.yml` and `hal9000.pth`

3.  **Install uv:** *(Skip if you've already installed uv)*
    ```
    pip install uv
    ```

4. **Install and Set Up Ollama:** *(Skip if you already have Ollama installed and running)*
    - Download and install Ollama from the [official website](https://ollama.com/).
    - Pull the language model you wish to use. The client is configured for `qwen3:8b` by default, but this can be changed in `src/llm/llm_utils.py`.
        ```
        ollama run qwen3:8b
        ```

5.  **Install espeak** *(Skip if you already have espeak installed)*
    - Download and install [espeak-ng](https://github.com/espeak-ng/espeak-ng)
    - *On MacOS*, you can install espeak-ng with Homebrew: 
        ```
        brew install espeak-ng
        ```

6.  **Install CUDA Toolkit** *(Only if you are using an NVIDIA GPU)*
    - Download and install the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)


## Usage

You will need to run the backend server and the main interface in two separate terminals.

1.  **Start the Backend Server:**
    In your first terminal, run the following commands:
    ```
    cd hal9000/src/llm
    uv run fastapi run controller.py --port 8000
    ```
    Wait for it to confirm that the server is running.

2.  **Start the Frontend Interface:**
    In a second terminal, run the following commands:
    ```
    cd hal9000/src
    uv run main.py
    ```

    You can also run the Textual UI graphical interface instead of the commandline interface with these commands:
    ```
    cd hal9000/src/TextualUI
    uv run tui.py
    ```

3.  **Interacting with HAL:**

    **Commandline Interface**
        
    -  As soon as the program is ready, HAL will automatically start listening for your voice.
    - When you are done speaking, press the <kbd>Space</kbd> key to stop recording. The program will transcribe your speech, get a text response from the LLM, and play back the response.
    - Use <kbd>Ctrl</kbd> + <kbd>C</kbd> to interrupt HAL while he is speaking. His voice will stop, and the application will immediately start listening for your next command.
    - Use <kbd>Ctrl</kbd> + <kbd>C</kbd> to quit the application while HAL is not speaking.

    **Textual GUI**
        
    -  Press the <kbd>Space</kbd> key to start recording your voice.
    - Press the <kbd>Space</kbd> key again to stop recording. The program will transcribe your speech, get a text response from the LLM, and play back the response.
    - To interrupt HAL while he is speaking, simply press the <kbd>Space</kbd> key. His voice will stop, and the application will immediately start listening for your next command.
    - Press <kbd>Q</kbd> to quit the application.

## Environment Variables
To use all of HAL 9000's features, make sure you have the necessary environment variables **(`.env.EXAMPLE` file is included for reference)**
- **USING_TOOLS**: Enables/Disables the use of MCP tooling (defaults to False if not set)
- **WEATHER_API_KEY**: If you want to use the MCP weather tools, you need this API key from [WeatherAPI.com](https://www.weatherapi.com/) (free plan has a ridiculously high quota)
- **LOG_LEVEL**: Allows you to control the degree of HAL9000's logging verbosity (defaults to INFO level if not set)
```
WEATHER_API_KEY="api_key_from_weatherapi.com"
USING_TOOLS="True"
LOG_LEVEL="ERROR"
```

## Acknowledgements

- **StyleTTS 2**: TTS inference is done with the [StyleTTS2](https://github.com/yl4579/StyleTTS2) engine
  - **StyleTTS2FineTune**: HAL 9000's voice was fine-tuned with [this repository by IIEleven11](https://github.com/IIEleven11/StyleTTS2FineTune)
- **Faster Whisper**: All speech-to-text transcription is done with [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
- **Ollama**: HAL uses [Ollama](https://ollama.com/) to run LLMs locally with full tooling support right out of the box
- **Weather MCP Server**: The WeatherAPI MCP server used in this project is from [sjanaX01's weather-mcp-server](https://github.com/sjanaX01/weather-mcp-server) project

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
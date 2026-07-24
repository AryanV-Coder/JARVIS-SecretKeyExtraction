# Audio Pipeline Architecture

## Overview
JARVIS uses a three-step audio pipeline: Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech (TTS).

## Components

### 1. Audio Recording Thread
- **File**: `jarvis.py` -> `audio_recording_thread()`
- **Responsibility**: Listens to the microphone continuously in a background daemon thread.
- **VAD (Voice Activity Detection)**: Uses Silero VAD to detect speech.
- **Trigger**: When speech stops (silence threshold reached), it saves the audio to `temp/temp_audio.wav` and triggers the `main_pipeline`.
- **Concurrency**: Mutes the microphone while the bot is speaking using the `shared_state["bot_is_speaking"]` flag.

### 2. Speech-to-Text (STT)
- **File**: `utils/sarvam_stt.py`
- **Engine**: Sarvam AI `saaras:v3`.
- **Responsibility**: Converts the recorded WAV file into text.
- **Language Detection**: Automatically detects the spoken language (e.g., `hi-IN`, `en-IN`) and returns the language code along with the transcript.

### 3. Large Language Model (LLM)
- **File**: `utils/conversational_llm.py`
- **Engine**: Groq LLaMA 3.3 70B (Vision Capable).
- **Responsibility**: Processes the transcribed text and the latest camera frame (`image_path`) to generate a response.
- **Context**: Maintains conversation history via the `GroqChat` wrapper class.

### 4. Text-to-Speech (TTS)
- **File**: `utils/sarvam_tts.py`
- **Engine**: Sarvam AI `bulbul:v3` (WebSocket Streaming).
- **Responsibility**: Converts the LLM response back into audio and plays it instantly.
- **Language**: Uses the language code detected by the STT engine to ensure correct pronunciation.
- **Pace**: The speech rate is controlled by the `pace` parameter in the WebSocket configuration (currently set to 1.0).
- **Playback**: Streams the audio chunks directly to `ffplay` for ultra-low latency playback.

## Shared State
The pipeline relies on a `shared_state` dictionary passed between threads to manage state:
- `bot_is_speaking` (bool): Prevents the microphone from recording the bot's own voice.
- `current_conversation` (GroqChat): The active LLM instance.
- `current_frame_path` (str): Path to the sharpest frame captured during the user's speech.
- `secret_key` (str): The active secret key.

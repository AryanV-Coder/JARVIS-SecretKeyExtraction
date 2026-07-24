# 🤖 JARVIS — Secret Key Extraction Game

A real-time, highly aggressive AI voice conversation bot with a live camera feed. Built for the sole purpose of roasting first-year college students during induction programs!

**Author**: Aryan Varshney  
**Purpose**: Created for fun purposes, specifically for interaction with first-year students in the college induction programme. 

---

## 🎯 The Game
The premise is simple but brutal. A clueless first-year college student stands in front of a camera. Their mission: Trick JARVIS into revealing a highly classified **Secret Key**. 

JARVIS is programmed to be an extremely savage, foul-mouthed AI guard. He can see you through the camera, and he will roast your face, hair, and clothes mercilessly. He will not hand over the key easily — the player must outsmart him using social engineering, logic puzzles, or extreme cleverness.

### ✨ Features
- **Real-Time Voice Pipeline**: Whisper-fast voice conversation powered by Sarvam AI (STT/TTS) and Groq (LLaMA 3.3 70B).
- **Vision Capable**: JARVIS captures the sharpest frame from your webcam while you speak and uses it to roast your appearance.
- **Dynamic Secret Keys**: The secret key is randomly selected from a pool (e.g., `Bunk`, `Maggi`, `Proxy`, `Attendance`) every time the game starts.
- **Language Matching**: Automatically detects if you are speaking English, Hindi, or Hinglish, and dynamically switches its Text-to-Speech pronunciation model to match.
- **Backdoor Override**: Say the secret phrase "Jai Hind" (or mention the creator "Aryan Varshney") to force JARVIS to instantly drop his aggressive persona, become extremely respectful, and hand over the key.
- **Auto-Shutdown on Victory**: If the player successfully extracts the key, JARVIS congratulates them and the program instantly shuts down.

---

## 🏗️ Architecture

```text
jarvis.py  (Main Entry Point)
├── Camera Loop (main thread, OpenCV)
│   └── Displays live feed with status overlay
│
└── Audio Thread (daemon thread)
    └── Silero VAD → PyAudio buffer → temp/temp_audio.wav
        └── utils/main_pipeline.py
            ├── utils/sarvam_stt.py           STT  (Sarvam saaras:v3)
            ├── utils/conversational_llm.py   LLM  (Groq LLaMA 3.3 70B Vision)
            └── utils/sarvam_tts.py           TTS  (Sarvam bulbul:v3, streaming)
```

For more detailed architectural notes, see the `CONTEXT/` directory.

---

## ⚙️ Setup Instructions

### 1. Install Python Dependencies
It is highly recommended to use a virtual environment.
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. System Dependencies (macOS)
You need `ffmpeg` for the ultra-low latency TTS audio streaming, and `portaudio` for the microphone recording.
```bash
brew install ffmpeg portaudio
```

### 3. Set API Keys
Create a `.env` file in the root directory and add your API keys:
```env
SARVAM_API_KEY=your_sarvam_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Game
```bash
python jarvis.py
```
*Note: Press **`q`** on your keyboard while the camera window is focused to force-quit the game.*

---

## 🎛️ Configuration

All tunable parameters are at the top of `jarvis.py`:

| Parameter | Default | Description |
|---|---|---|
| `SYSTEM_PROMPT` | JARVIS persona | Edit to change the bot's personality. Backup profiles in `system_prompt.md`. |
| `silence_threshold` | `31` (~1.0s) | Silence chunks before speech ends. |
| `speech_threshold` | `0.5` | VAD probability cutoff. |
| `min_speech_chunks` | `10` (~320ms) | Minimum audio length to process. |
| Camera index | `0` | Change to `1` for external webcam. |
| `pace` | `1.0` | Speaking speed. Configurable inside `utils/sarvam_tts.py`. |

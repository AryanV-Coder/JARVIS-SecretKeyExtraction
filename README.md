# 🤖 JARVIS — Voice Conversation Bot

A real-time AI voice conversation bot with a live camera feed. Built on the same architecture as **Vayu** but without face recognition — just camera + voice pipeline.

---

## 🏗️ Architecture

```
jarvis.py  (Main Entry Point)
├── Camera Loop (main thread, OpenCV)
│   └── Displays live feed with status overlay
│
└── Audio Thread (daemon thread)
    └── Silero VAD → PyAudio buffer → temp/temp_audio.wav
        └── utils/main_pipeline.py
            ├── utils/sarvam_stt.py       STT  (Sarvam saaras:v3)
            ├── utils/conversational_llm.py   LLM  (Groq LLaMA 3.3 70B)
            └── utils/sarvam_tts.py       TTS  (Sarvam bulbul:v3, streaming)

shared_state dict  ←  thread-safe communication bus
  ├── current_conversation  Active ConversationalLLM instance
  ├── bot_is_speaking       Mic mute flag during playback
  └── running               Shutdown signal
```

---

## 📁 Project Structure

```
JARVIS_SecretKeyExtraction/
├── jarvis.py               🚀 Main entry point — camera + audio loop
├── requirements.txt
├── .env                    API keys
│
└── utils/
    ├── main_pipeline.py    🔗 STT → LLM → TTS orchestrator
    ├── sarvam_stt.py       🎙️ Sarvam saaras:v3 Speech-to-Text
    ├── sarvam_tts.py       🔊 Sarvam bulbul:v3 Text-to-Speech (streaming)
    └── conversational_llm.py  🧠 Groq LLaMA 3.3 70B chat
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. System dependencies (macOS)

```bash
brew install ffmpeg portaudio
```

### 3. Set API keys

Edit `.env`:
```env
SARVAM_API_KEY=your_sarvam_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run

```bash
python jarvis.py
```

Press **`q`** to quit.

---

## 🎛️ Configuration

All tunable parameters are at the top of `jarvis.py`:

| Parameter | Default | Description |
|---|---|---|
| `SYSTEM_PROMPT` | JARVIS persona | Edit to change the bot's personality |
| `silence_threshold` | `31` (~1.0s) | Silence chunks before speech ends |
| `speech_threshold` | `0.5` | VAD probability cutoff |
| `min_speech_chunks` | `10` (~320ms) | Minimum audio length to process |
| Camera index | `0` | Change to `1` for external webcam |

---

## 🔄 Switching STT Engine

In `utils/main_pipeline.py`, swap the import:

```python
# Cloud (default) — Hindi/Hinglish support:
from utils.sarvam_stt import stt

# Local offline (Whisper):
# from utils.whisper_stt import stt
```

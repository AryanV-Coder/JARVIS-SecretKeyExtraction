import os
from dotenv import load_dotenv
from utils.sarvam_tts import speak_text
from utils.sarvam_stt import stt

# To switch to offline Whisper STT instead of Sarvam, comment the line above and uncomment:
# from utils.whisper_stt import stt

load_dotenv()


def main_pipeline(audio_path, shared_state):
    """
    Process recorded audio through the full voice pipeline:
    STT (Sarvam saaras:v3) → LLM (Groq LLaMA) → TTS (Sarvam bulbul:v3)

    Args:
        audio_path: Path to the recorded audio WAV file.
        shared_state: Shared state dict with 'current_conversation' and 'bot_is_speaking'.
    """
    conversation = shared_state.get("current_conversation")

    if not conversation:
        print("⚠️  No conversation context set in shared_state.")
        return

    # Step 1: STT — Sarvam saaras:v3
    try:
        user_text = stt(audio_path)
    except Exception as e:
        print(f"❌ STT Error: {e}")
        return

    if not user_text or not user_text.strip():
        print("⚠️  No transcription detected, skipping.")
        return

    print(f"👤 User: {user_text}")

    # Step 2: LLM — Groq LLaMA 3.3 70B
    try:
        print("🤖 Thinking...")
        response = conversation.send_message(user_text)
        print(f"🤖 Bot: {response}")
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return

    # Step 3: TTS — Sarvam bulbul:v3 (streaming)
    speak_text(response, shared_state)

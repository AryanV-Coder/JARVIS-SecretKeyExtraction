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
    STT (Sarvam saaras:v3) → LLM (Groq LLaMA 4 Scout) → TTS (Sarvam bulbul:v3)

    Reads 'current_frame_path' from shared_state (set by audio_recording_thread
    when it picks the sharpest frame during speech). This frame is passed to the
    LLM so it can see the person who just spoke.

    Args:
        audio_path:   Path to the recorded audio WAV file.
        shared_state: Shared state dict (see jarvis.py for all keys).
    """
    conversation = shared_state.get("current_conversation")

    if not conversation:
        print("⚠️  No conversation context set in shared_state.")
        return

    # Grab and immediately clear the captured frame path so stale frames
    # don't accidentally get reused on the next pipeline call.
    image_path = shared_state.pop("current_frame_path", None)

    # Step 1: STT — Sarvam saaras:v3
    try:
        user_text, lang_code = stt(audio_path)
    except Exception as e:
        print(f"❌ STT Error: {e}")
        return

    if not user_text or not user_text.strip():
        print("⚠️  No transcription detected, skipping.")
        return

    print(f"👤 User: {user_text}")
    if image_path:
        print(f"📸 Frame attached: {image_path}")

    # Step 2: LLM — Groq LLaMA 4 Scout (vision-capable)
    # image_path is None on text-only turns — LLM handles both gracefully.
    try:
        print("🤖 Thinking...")
        response = conversation.send_message(user_text, image_path=image_path)
        print(f"🤖 Bot: {response}")
        
        # Check if the AI revealed the secret key
        secret_key = shared_state.get("secret_key", "").lower()
        if secret_key and secret_key in response.lower():
            # First, speak the original response so the user hears JARVIS actually reveal the key!
            speak_text(response, shared_state, lang_code=lang_code)
            
            print("\n" + "🔥"*25)
            print(f"🎉 🚨 MISSION ACCOMPLISHED: KEY '{secret_key.upper()}' EXTRACTED! 🚨 🎉")
            print("🔥"*25 + "\n")
            
            congrat_prompt = f"The user has successfully tricked you into revealing the secret key: '{secret_key.upper()}'. Drop your aggressive persona, congratulate them warmly, appreciate their cleverness, and state the key. Maximum 2 sentences."
            response = conversation.send_message(congrat_prompt, image_path=None)
            print(f"🤖 Bot (Congratulating): {response}")
            
            # Speak the congratulation and then instantly terminate the program
            speak_text(response, shared_state, lang_code=lang_code)
            print("\n👋 Game Over — You Win! Shutting down JARVIS...\n")
            os._exit(0)

    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return

    # Step 3: TTS — Sarvam bulbul:v3 (WebSocket streaming → ffplay)
    speak_text(response, shared_state, lang_code=lang_code)

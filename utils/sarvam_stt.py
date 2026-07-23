import os
import sys
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError
from dotenv import load_dotenv

load_dotenv()


def stt(audio_path):
    """
    Transcribe audio from a WAV file using Sarvam AI's saaras:v3 model.
    Supports Hindi, English, and Hinglish.

    Args:
        audio_path: Path to the recorded WAV file.

    Returns:
        Transcribed text string, or "" on failure.
    """
    client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

    try:
        response = client.speech_to_text.transcribe(
            file=open(audio_path, "rb"),
            model="saaras:v3",
            mode="transcribe",
        )

        lang = response.language_code if hasattr(response, "language_code") and response.language_code else "unknown"
        print(f"[Sarvam STT] Detected language: {lang}")

        return response.transcript

    except ApiError as e:
        if e.status_code == 429:
            print("\n🚨 🚨 🚨 [CRITICAL ERROR] 🚨 🚨 🚨")
            print("🛑 RATE LIMIT EXCEEDED: Sarvam AI STT API")
            print("👉 Too many speech-to-text requests. Please wait.")
            print("Shutting down safely...")
            print("🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨\n")
            os._exit(1)
        else:
            print(f"❌ Sarvam STT API Error ({e.status_code}): {e.body}")
            return ""

    except Exception as e:
        print(f"❌ Unexpected Sarvam STT Error: {e}")
        return ""

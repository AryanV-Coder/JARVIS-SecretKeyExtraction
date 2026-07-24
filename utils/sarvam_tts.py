import asyncio
import base64
import os
import sys
import time
import subprocess
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse
from sarvamai.core.api_error import ApiError

load_dotenv()


async def _tts_stream_and_play(text, lang_code):
    """
    Convert text to speech using Sarvam bulbul:v3 and stream directly into ffplay.
    Audio playback starts within ~0.4s of the first chunk arriving.
    """
    client = AsyncSarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

    # Spawn ffplay reading from stdin — audio starts as soon as the first chunk arrives
    player_process = subprocess.Popen(
        ["ffplay", "-autoexit", "-nodisp", "-i", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        async with client.text_to_speech_streaming.connect(
            model="bulbul:v3", send_completion_event=True
        ) as ws:
            await ws.configure(
                target_language_code=lang_code,
                speaker="shubh",
                pace=1.0,  # 1.0 is default, lower is slower
            )
            print("🔊 TTS: Configuration sent")

            await ws.convert(text)
            print("🔊 TTS: Text sent, waiting for audio chunks...")
            await ws.flush()

            chunk_count = 0
            async for message in ws:
                if isinstance(message, AudioOutput):
                    chunk_count += 1
                    audio_chunk = base64.b64decode(message.data.audio)

                    if player_process.stdin:
                        player_process.stdin.write(audio_chunk)
                        player_process.stdin.flush()

                elif isinstance(message, EventResponse):
                    print(f"🔊 TTS: Event — {message.data.event_type}")
                    if message.data.event_type == "final":
                        break

            print(f"✅ TTS: Received {chunk_count} audio chunks.")

            if hasattr(ws, "_websocket") and not ws._websocket.closed:
                await ws._websocket.close()

    except ApiError as e:
        if e.status_code == 429:
            print("\n🚨 🚨 🚨 [CRITICAL ERROR] 🚨 🚨 🚨")
            print("🛑 RATE LIMIT EXCEEDED: Sarvam AI TTS API")
            print("👉 Too many text-to-speech requests. Please wait.")
            print("Shutting down safely...")
            print("🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨\n")
            os._exit(1)
        else:
            print(f"❌ Sarvam TTS API Error ({e.status_code}): {e.body}")

    except Exception as e:
        print(f"❌ Unexpected Sarvam TTS Error: {e}")

    finally:
        if player_process.stdin:
            player_process.stdin.close()

        print("🔊 Waiting for audio playback to finish...")
        player_process.wait()
        print("✅ Audio playback complete")


def speak_text(text, shared_state, lang_code="hi-IN"):
    """
    Convert text to speech and play it via streaming.
    Mutes the microphone (via shared_state flag) during playback.

    Args:
        text: The text to speak.
        shared_state: Shared dict with 'bot_is_speaking' flag.
    """
    shared_state["bot_is_speaking"] = True
    try:
        print(f"🔊 Speaking: \"{text}\" [Lang: {lang_code}]")
        asyncio.run(_tts_stream_and_play(text, lang_code))
    except Exception as e:
        print(f"❌ TTS Error: {e}")
    finally:
        time.sleep(0.2)  # Small buffer before re-enabling mic
        shared_state["bot_is_speaking"] = False

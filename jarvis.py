import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import time
import threading
import pyaudio
import numpy as np
import wave
from utils.conversational_llm import ConversationalLLM
from utils.main_pipeline import main_pipeline


# ============ AUDIO CONFIGURATION ============
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512  # 32ms chunks at 16kHz

# Defer VAD model load — avoids unnecessary init overhead
vad_model = None


def get_vad_model():
    global vad_model
    if vad_model is None:
        import torch
        print("Loading Silero VAD model...")
        vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        print("✅ VAD model loaded")
    return vad_model


# ============ SHARED STATE ============
# Central communication bus between the camera thread and audio thread.
# Mirrors Vayu's shared_state pattern exactly.
shared_state = {
    "current_conversation": None,  # Active ConversationalLLM instance
    "bot_is_speaking": False,       # Mic is muted while True
    "running": True,                # Set to False to stop all threads
}


# ============ SYSTEM PROMPT ============
# Edit this to change JARVIS's personality.
SYSTEM_PROMPT = (
    "You are J.A.R.V.I.S., a witty, charming, and highly intelligent AI assistant. "
    "You are having a live voice conversation with a person. "
    "--- CRITICAL RULES ---\n"
    "1. Mirror the user's language exactly (Hindi, English, or Hinglish).\n"
    "2. Keep responses concise — maximum 2 short sentences.\n"
    "3. Never use emojis or special characters — generate spoken text only.\n"
    "4. Be warm, engaging, and conversational.\n"
    "5. If they troll or insult you, playfully roast them back."
)


# ============ AUDIO RECORDING THREAD ============

def audio_recording_thread():
    """
    Daemon thread: Continuously records audio from the microphone.
    Uses Silero VAD to detect when the user speaks and when they stop.
    On silence detection, saves the buffered audio to a WAV file and
    triggers the main_pipeline (STT → LLM → TTS).

    Mirrors Vayu's audio_recording_thread() exactly.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("🎙️  Audio recording started — listening...")

    is_recording = False
    audio_buffer = []
    silence_counter = 0
    silence_threshold = 31    # ~1.0s of silence ends the recording
    speech_threshold = 0.5    # VAD probability cutoff
    min_speech_chunks = 10    # ~320ms minimum to be considered valid speech

    try:
        while shared_state["running"]:
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)

            # Discard input while bot is speaking (prevents echo/feedback)
            if shared_state["bot_is_speaking"]:
                if is_recording:
                    is_recording = False
                    audio_buffer = []
                    silence_counter = 0
                continue

            # Run Silero VAD on the current chunk
            audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            model = get_vad_model()
            import torch
            speech_prob = model(torch.from_numpy(audio_float32), RATE).item()
            is_speech = speech_prob > speech_threshold

            if is_speech:
                if not is_recording:
                    print("\n🎤 Speech detected — recording...")
                    is_recording = True
                audio_buffer.append(audio_chunk)
                silence_counter = 0

            elif is_recording:
                audio_buffer.append(audio_chunk)
                silence_counter += 1

                if silence_counter >= silence_threshold:
                    print("🔇 Silence detected — processing...")

                    if len(audio_buffer) >= min_speech_chunks:
                        # Save buffered audio to a temp WAV file
                        audio_path = os.path.join("temp", "temp_audio.wav")
                        os.makedirs("temp", exist_ok=True)

                        with wave.open(audio_path, "wb") as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(2)   # 16-bit = 2 bytes
                            wf.setframerate(RATE)
                            wf.writeframes(b"".join(audio_buffer))

                        # Fire the STT → LLM → TTS pipeline (blocks this thread)
                        main_pipeline(audio_path, shared_state)
                    else:
                        print("⚠️  Audio too short, skipping...")

                    # Reset recording state
                    is_recording = False
                    audio_buffer = []
                    silence_counter = 0

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("🎙️  Audio recording stopped")


# ============ CAMERA + MAIN LOOP ============

def start():
    """
    Main function:
    - Opens the camera (display-only, no face recognition).
    - Starts the audio recording daemon thread.
    - Initializes a ConversationalLLM with the configured system prompt.
    - Runs the camera display loop until 'q' is pressed.

    Mirrors Vayu's start_live_recognition() — same structure, no process pool.
    """

    # Initialize the conversation context
    conversation = ConversationalLLM(SYSTEM_PROMPT)
    shared_state["current_conversation"] = conversation
    print("✅ ConversationalLLM initialized")

    # Open camera
    cap = cv2.VideoCapture(0)
    # If your built-in camera takes priority over an external webcam, change 0 → 1

    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return

    print("📹 Camera opened successfully")
    print("Press 'q' to quit\n")

    # Start audio daemon thread
    audio_thread = threading.Thread(target=audio_recording_thread, daemon=True)
    audio_thread.start()

    # ──────────────── Camera Display Loop ────────────────
    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Error: Failed to capture frame")
            break

        # Overlay status indicator
        status = "Listening..." if not shared_state["bot_is_speaking"] else "Speaking..."
        color = (0, 255, 0) if not shared_state["bot_is_speaking"] else (0, 165, 255)
        cv2.putText(frame, status, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("J.A.R.V.I.S.", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting...")
            break

    # ──────────────── Cleanup ────────────────
    shared_state["running"] = False
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed")


if __name__ == "__main__":
    start()

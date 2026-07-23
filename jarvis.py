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

# ============ FRAME CONFIGURATION ============
# Frames are captured during speech and the sharpest one is sent to the LLM.
FRAME_MAX_WIDTH = 512          # Resize to this width before encoding (keeps JPEG small)
FRAME_JPEG_QUALITY = 80        # JPEG quality (80 = good quality, small size ~60-100KB)
FRAME_COLLECT_EVERY_N = 15     # Collect a frame every N audio chunks (~480ms at 32ms/chunk)


# ============ SHARED STATE ============
# Central communication bus between the camera thread and the audio thread.
# Mirrors Vayu's shared_state pattern exactly.
#
# Keys:
#   current_conversation  — Active ConversationalLLM instance
#   bot_is_speaking       — True while TTS is playing (mic is muted)
#   running               — Set to False to stop all threads
#   latest_frame          — Most recent camera frame (numpy array), updated every loop tick
#   current_frame_path    — Path to the best frame saved from the last speech segment;
#                           written by audio thread, consumed & cleared by main_pipeline
shared_state = {
    "current_conversation": None,
    "bot_is_speaking": False,
    "running": True,
    "latest_frame": None,
    "current_frame_path": None,
}


# ============ SYSTEM PROMPT ============
# Edit this to change JARVIS's personality.
SYSTEM_PROMPT = (
    "You are J.A.R.V.I.S., a witty, charming, and highly intelligent AI assistant. "
    "You are having a live voice conversation with a person standing in front of a camera. "
    "You may receive an image of the person alongside their words — use it to make the "
    "conversation more personal and engaging (e.g., comment on what they're wearing, "
    "their expression, or anything visually interesting).\n"
    "--- CRITICAL RULES ---\n"
    "1. Mirror the user's language exactly (Hindi, English, or Hinglish).\n"
    "2. Keep responses concise — maximum 2 short sentences.\n"
    "3. Never use emojis or special characters — generate spoken text only.\n"
    "4. Be warm, engaging, and conversational.\n"
    "5. If they troll or insult you, playfully roast them back.\n"
    "6. Output ONLY your final spoken response. Do not include reasoning, thinking steps, "
    "or any internal monologue. /no_think"
)


# ============ VAD MODEL (lazy-loaded) ============
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


# ============ FRAME UTILITIES ============

def _sharpness(frame: np.ndarray) -> float:
    """
    Compute the sharpness of a frame using the Laplacian variance method.
    Higher = sharper. Used to pick the best frame from the speech window.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Resize frame to max_width while preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def pick_and_save_best_frame(frames: list, save_dir: str) -> str | None:
    """
    From a list of frames collected during a speech segment, pick the
    sharpest one, resize it, compress as JPEG, and save to disk.

    Args:
        frames:   List of numpy BGR frames captured during speech.
        save_dir: Directory to save the JPEG file in.

    Returns:
        Absolute path to the saved JPEG, or None if frames list is empty.
    """
    if not frames:
        print("⚠️  No frames collected during speech, skipping vision.")
        return None

    # Pick sharpest frame
    best_frame = max(frames, key=_sharpness)

    # Resize to keep JPEG payload small
    best_frame = _resize_frame(best_frame, FRAME_MAX_WIDTH)

    # Save as JPEG
    os.makedirs(save_dir, exist_ok=True)
    frame_path = os.path.join(save_dir, "best_frame.jpg")
    cv2.imwrite(frame_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY])

    sharpness_score = _sharpness(best_frame)
    print(f"📸 Best frame saved: {frame_path} "
          f"(picked 1 of {len(frames)}, sharpness={sharpness_score:.1f})")
    return frame_path


# ============ AUDIO RECORDING THREAD ============

def audio_recording_thread():
    """
    Daemon thread: Continuously records audio from the microphone.
    Uses Silero VAD to detect when the user speaks and when they stop.

    Frame collection (Option 4):
      - On speech START  → immediately grab shared_state['latest_frame']
      - During speech    → grab a new frame every FRAME_COLLECT_EVERY_N chunks
      - On silence END   → call pick_and_save_best_frame(), store path in
                           shared_state['current_frame_path'] for main_pipeline

    Mirrors Vayu's audio_recording_thread() — same state machine, same thresholds.
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
    speech_frames = []          # Camera frames collected during this speech segment
    frame_chunk_counter = 0     # Counts chunks since last frame grab
    silence_counter = 0
    silence_threshold = 31      # ~1.0s of silence ends the recording
    speech_threshold = 0.5      # VAD probability cutoff
    min_speech_chunks = 10      # ~320ms minimum to be considered valid speech

    try:
        while shared_state["running"]:
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)

            # Discard input while bot is speaking (prevents echo/feedback)
            if shared_state["bot_is_speaking"]:
                if is_recording:
                    is_recording = False
                    audio_buffer = []
                    speech_frames = []
                    frame_chunk_counter = 0
                    silence_counter = 0
                continue

            # ── Silero VAD ──────────────────────────────────────────────────
            audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            model = get_vad_model()
            import torch
            speech_prob = model(torch.from_numpy(audio_float32), RATE).item()
            is_speech = speech_prob > speech_threshold

            # ── Speech detected ─────────────────────────────────────────────
            if is_speech:
                if not is_recording:
                    print("\n🎤 Speech detected — recording...")
                    is_recording = True
                    speech_frames = []
                    frame_chunk_counter = 0

                    # Grab the very first frame immediately on speech start
                    first_frame = shared_state.get("latest_frame")
                    if first_frame is not None:
                        speech_frames.append(first_frame.copy())

                audio_buffer.append(audio_chunk)
                silence_counter = 0

                # Periodically grab a new frame during speech
                frame_chunk_counter += 1
                if frame_chunk_counter % FRAME_COLLECT_EVERY_N == 0:
                    frame = shared_state.get("latest_frame")
                    if frame is not None:
                        speech_frames.append(frame.copy())

            # ── Silence after speech ────────────────────────────────────────
            elif is_recording:
                audio_buffer.append(audio_chunk)
                silence_counter += 1

                if silence_counter >= silence_threshold:
                    print("🔇 Silence detected — processing...")

                    if len(audio_buffer) >= min_speech_chunks:
                        # ── Pick best frame & store path for main_pipeline ──
                        frame_path = pick_and_save_best_frame(speech_frames, "temp")
                        shared_state["current_frame_path"] = frame_path

                        # ── Save WAV ────────────────────────────────────────
                        audio_path = os.path.join("temp", "temp_audio.wav")
                        os.makedirs("temp", exist_ok=True)

                        with wave.open(audio_path, "wb") as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(2)   # 16-bit = 2 bytes
                            wf.setframerate(RATE)
                            wf.writeframes(b"".join(audio_buffer))

                        # ── Fire STT → LLM → TTS ───────────────────────────
                        main_pipeline(audio_path, shared_state)

                    else:
                        print("⚠️  Audio too short, skipping...")

                    # Reset
                    is_recording = False
                    audio_buffer = []
                    speech_frames = []
                    frame_chunk_counter = 0
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
    - Opens the camera.
    - Every loop tick writes the latest frame into shared_state['latest_frame']
      so the audio thread can sample it during speech.
    - Starts the audio recording daemon thread.
    - Displays the live feed with a status overlay.
    - Press 'q' to quit.

    Mirrors Vayu's start_live_recognition() structure exactly.
    """

    # Initialize conversation
    conversation = ConversationalLLM(SYSTEM_PROMPT)
    shared_state["current_conversation"] = conversation
    print("✅ ConversationalLLM (Qwen — vision) initialized")

    # Open camera
    cap = cv2.VideoCapture(0)
    # Change 0 → 1 if your built-in camera takes priority over an external webcam

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

        # ── Publish latest frame to shared_state (read by audio thread) ──
        # .copy() ensures the audio thread gets a stable snapshot and not a
        # reference that the camera loop will overwrite on the next tick.
        shared_state["latest_frame"] = frame.copy()

        # ── Status overlay ────────────────────────────────────────────────
        is_speaking = shared_state["bot_is_speaking"]
        status = "Speaking..." if is_speaking else "Listening..."
        color  = (0, 165, 255) if is_speaking else (0, 255, 0)
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

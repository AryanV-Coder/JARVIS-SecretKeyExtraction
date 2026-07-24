import os
import re
import json
import base64
from groq import Groq, RateLimitError, APIStatusError
from dotenv import load_dotenv

load_dotenv()

# ============ MODEL ============
VISION_MODEL = "qwen/qwen3.6-27b"

# ============ FEW-SHOT EXAMPLES ============
# Path to your fineTuning.json file. Set to None to disable.
FINE_TUNING_FILE = None


def _encode_image(image_path: str) -> str:
    """Read an image file and return its base64-encoded JPEG string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _load_few_shot_examples(filepath: str) -> list:
    """
    Load few-shot examples from a JSON file and convert them into
    chat message pairs (user + assistant) ready to inject into history.

    Expected JSON format:
    [
      {
        "base64_image": "<raw base64 string, NO data:image prefix>",
        "user_text":    "<what the user said — can be empty>",
        "assistant_response": "<how JARVIS should respond>"
      },
      ...
    ]

    Returns a flat list of {role, content} dicts.
    """
    if not filepath or not os.path.exists(filepath):
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            examples = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️  fineTuning.json load error: {e}")
        return []

    # --- FEW-SHOT LIMIT ---
    # Groq has strict limits on the number of images per request (usually 3 max).
    # We take the top 2 examples here + 1 live camera frame = 3 images total.
    # To load ALL examples in the file instead, just comment out the line below:
    examples = examples[:2]

    messages = []
    for i, ex in enumerate(examples):
        b64    = ex.get("base64_image", "").strip()
        text   = ex.get("user_text", "").strip()
        answer = ex.get("assistant_response", "").strip()

        if not b64 or not answer:
            print(f"⚠️  fineTuning example #{i+1} missing image or response — skipped.")
            continue

        # Build user message: text first (may be empty), then image
        user_content = [
            {"type": "text", "text": text or "[image]"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        messages.append({"role": "user",      "content": user_content})
        messages.append({"role": "assistant",  "content": answer})

    return messages


class ConversationalLLM:
    """
    Stateful conversational LLM powered by Groq Qwen 3.6 27B (vision-capable).

    Each call to send_message() optionally accepts an image_path.
    When provided, the frame is embedded as a base64 JPEG alongside the
    user's text so the model can see the person who is speaking.

    Message format follows the official Groq vision docs exactly:
      content = [{type: text}, {type: image_url}]  ← text always first

    Chat history is preserved across turns. Only the user turn that
    includes an image carries the image — subsequent text-only turns
    are sent as plain strings, keeping token costs low.
    """

    def __init__(self, system_prompt: str, fine_tuning_file: str = FINE_TUNING_FILE):
        """
        Args:
            system_prompt:    System-level instruction for the assistant's persona.
            fine_tuning_file: Path to fineTuning.json. Pass None to skip.
        """
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.chat_history = [
            {"role": "system", "content": system_prompt}
        ]

        # ── Inject few-shot examples ──────────────────────────────────────
        # These are loaded once at startup and placed right after the system
        # prompt. The model treats them as prior conversation turns, learning
        # the desired response style from the image+text→response pairs.
        few_shot = _load_few_shot_examples(fine_tuning_file)
        if few_shot:
            self.chat_history.extend(few_shot)
            print(f"📚 Loaded {len(few_shot) // 2} few-shot example(s) from {os.path.basename(fine_tuning_file)}")
        else:
            print("📚 No few-shot examples loaded (file missing or empty).")

    def send_message(self, user_text: str, image_path: str = None) -> str:
        """
        Send a user message (optionally with a camera frame) and get a response.

        Args:
            user_text:   The user's transcribed speech.
            image_path:  Optional path to a JPEG frame captured during speech.
                         The image is sent ONLY for this turn and is NEVER stored
                         in chat_history — this prevents token explosion and the
                         'too many images' 400 error on subsequent turns.

        Returns:
            The assistant's response string.
        """

        # ── Build the current user message (may include image) ──────────────
        # Order follows official Groq vision docs: text first, image_url second.
        # https://console.groq.com/docs/vision
        if image_path and os.path.exists(image_path):
            b64 = _encode_image(image_path)
            current_user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
            print(f"📸 Vision: sending frame → {image_path}")
        else:
            current_user_message = {"role": "user", "content": user_text}
            if image_path:
                print(f"⚠️  Vision: frame not found ({image_path}), sending text only")

        # ── Build messages for the API call ─────────────────────────────────
        # IMPORTANT: We do NOT append to self.chat_history yet.
        # The image travels only in this single API call.
        # After the call we store only the plain text — this keeps the history
        # lean and avoids: 413 token overflow, 400 "too many images" errors.
        messages_for_api = self.chat_history + [current_user_message]

        # ── LLM call ────────────────────────────────────────────────────────
        try:
            response = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages_for_api,
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
                reasoning_effort="none",  # Groq's official param to disable Qwen thinking
            )

            raw = response.choices[0].message.content or ""

            # ── Strip Qwen reasoning blocks ──────────────────────────────────
            # /no_think per-turn suppresses thinking, but strip as safety net.
            # Also handles unclosed <think> tags (model cut off mid-thought).
            answer = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
            answer = re.sub(r'<think>.*',          '',    answer, flags=re.DOTALL).strip()

            if not answer:
                # Fallback: if stripping removed everything, something is wrong
                print("⚠️  LLM returned only a think block with no final answer.")
                answer = "Sorry, I got a bit lost in thought there. Could you repeat that?"

            # ── Persist to history (text only, no image) ─────────────────────
            self.chat_history.append({"role": "user", "content": user_text})
            self.chat_history.append({"role": "assistant", "content": answer})
            return answer

        except RateLimitError:
            print("\n🚨 🚨 🚨 [CRITICAL ERROR] 🚨 🚨 🚨")
            print("🛑 RATE LIMIT EXCEEDED: Groq LLM API")
            print("👉 Too many LLM requests. Please wait.")
            print("Shutting down safely...")
            print("🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨\n")
            os._exit(1)

        except APIStatusError as e:
            print(f"❌ Groq API Error ({e.status_code}): {e.message}")
            return "I'm having a little trouble thinking right now. Please try again!"

        except Exception as e:
            print(f"❌ Unexpected Groq Error: {e}")
            return "Oops, something went wrong in my brain circuit!"

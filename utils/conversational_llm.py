import os
import re
import base64
from groq import Groq, RateLimitError, APIStatusError
from dotenv import load_dotenv

load_dotenv()

# ============ MODEL ============
VISION_MODEL = "qwen/qwen3.6-27b"


def _encode_image(image_path: str) -> str:
    """Read an image file and return its base64-encoded JPEG string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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

    def __init__(self, system_prompt: str):
        """
        Args:
            system_prompt: System-level instruction for the assistant's persona.
        """
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.chat_history = [
            {"role": "system", "content": system_prompt}
        ]

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

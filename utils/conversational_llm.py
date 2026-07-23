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
            image_path:  Optional path to a JPEG frame of the user captured
                         during their speech. When provided, sent to LLaMA 4 Scout
                         as a base64 image_url block.

        Returns:
            The assistant's response string.
        """

        # ── Build user content ──────────────────────────────────────────────
        # Order follows official Groq vision docs: text block first, image_url second.
        # https://console.groq.com/docs/vision
        if image_path and os.path.exists(image_path):
            b64 = _encode_image(image_path)
            user_content = [
                {
                    "type": "text",
                    "text": user_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                    },
                },
            ]
            print(f"📸 Vision: sending frame → {image_path}")
        else:
            # Text-only turn — no image available or not provided
            user_content = user_text
            if image_path:
                print(f"⚠️  Vision: frame path given but file not found ({image_path}), sending text only")

        self.chat_history.append({"role": "user", "content": user_content})

        # ── LLM call ────────────────────────────────────────────────────────
        try:
            response = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=self.chat_history,
                temperature=0.7,
                max_completion_tokens=4096,  # Qwen reasoning uses tokens before the real answer
                top_p=1,
                stream=False,
                stop=None,
            )

            raw = response.choices[0].message.content or ""

            # ── Strip Qwen reasoning blocks ──────────────────────────────────
            # Qwen 3.6 is a reasoning model: it wraps its chain-of-thought in
            # <think>...</think> before the actual answer. We must strip this
            # before logging or passing to TTS, otherwise TTS gets thousands
            # of tokens of internal monologue and returns 0 audio chunks.
            answer = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

            if not answer:
                # Fallback: if stripping removed everything, something is wrong
                print("⚠️  LLM returned only a think block with no final answer.")
                answer = "Sorry, I got a bit lost in thought there. Could you repeat that?"

            # Store clean answer in history (no think block, no image in assistant turn)
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

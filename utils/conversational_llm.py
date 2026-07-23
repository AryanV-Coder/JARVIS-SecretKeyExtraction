import os
import sys
from groq import Groq, RateLimitError, APIStatusError
from dotenv import load_dotenv

load_dotenv()


class ConversationalLLM:
    """
    A stateful conversational LLM powered by Groq (LLaMA 3.3 70B).

    Maintains full chat history per instance. Create a new instance
    to reset the conversation context.
    """

    def __init__(self, system_prompt: str):
        """
        Initialize the conversational LLM with a given system prompt.

        Args:
            system_prompt: The system-level instruction for the assistant's persona.
        """
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.chat_history = [
            {"role": "system", "content": system_prompt}
        ]

    def send_message(self, user_text: str) -> str:
        """
        Send a user message and get a response.

        Args:
            user_text: The user's transcribed speech.

        Returns:
            The assistant's response string.
        """
        self.chat_history.append({"role": "user", "content": user_text})

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=self.chat_history,
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            answer = response.choices[0].message.content
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

# Secret Key Extraction Game Mechanics

## Overview
JARVIS acts as an aggressive AI guard protecting a dynamically generated secret key. The user has to trick JARVIS into revealing the key using social engineering, logic puzzles, or the backdoor.

## Core Features
1. **Dynamic Secret Key**: A secret key is randomly selected from a predefined list at startup.
2. **Backdoor Override**: The user can force JARVIS to reveal the key by saying the secret phrase "Jai Hind".
3. **Victory Condition**: If the secret key is present in JARVIS's response, the game ends immediately.
4. **Victory Flow**:
   - JARVIS speaks the response containing the key.
   - The system detects the key and prints a victory banner.
   - A new congratulatory prompt is sent to the LLM to generate a respectful congratulations.
   - JARVIS speaks the congratulations.
   - The application shuts down (`os._exit(0)`).

## Implementation Details
- **Key Injection**: The selected key is injected into the `SYSTEM_PROMPT` in `jarvis.py` at runtime.
- **Victory Detection**: Handled in `utils/main_pipeline.py`. It checks `if secret_key in response.lower():`.
- **System Prompts**: The prompts instruct JARVIS to never reveal the key directly and to roast the user mercilessly. The prompts include a `VARIETY` rule to prevent repetitive responses.

## Key Files
- `jarvis.py`: Startup logic, key selection, welcome speech, and `SYSTEM_PROMPT`.
- `utils/main_pipeline.py`: Pipeline execution and victory detection logic.
- `system_prompt.md`: Backup/reference file for Intense and Light roasting profiles.

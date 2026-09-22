import os

_PROMPTS_DIR = os.path.dirname(__file__)

with open(os.path.join(_PROMPTS_DIR), "system.md") as f:
    SYSTEM_PROMPT = f.read()

def load_system_prompt():
    return SYSTEM_PROMPT
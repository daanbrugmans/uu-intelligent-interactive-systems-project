import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "./checkpoints/best_head_ls.pt")
    CAMERA_ID = int(os.getenv("CAMERA_ID", "0"))
    
    EMOTION_THRESHOLDS = {
        "valence": 7.01,
        "arousal": 4.85
    }
    
    MAX_CONVERSATION_TIME = 120
    MAX_CONVERSATION_ROUNDS = 10
    FRAME_SKIP = 5
    
    @classmethod
    def validate(cls):
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env")
        return True

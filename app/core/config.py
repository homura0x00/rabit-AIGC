
# Define environment types
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()

class Environment(str, Enum):
    """Application environment types.

    Defines the possible environments the application can run in:
    development, staging, production, and test.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"

# Determine environment
def get_environment() -> Environment:
    """Get the current environment.

    Returns:
        Environment: The current environment (development, staging, production, or test)
    """
    match os.getenv("APP_ENV", "development").lower():
        case "production" | "prod":
            return Environment.PRODUCTION
        case "staging" | "stage":
            return Environment.STAGING
        case "test":
            return Environment.TEST
        case _:
            return Environment.DEVELOPMENT
    

class Settings:
    def __init__(self) -> None:
        # Set the environment
        self.ENVIRONMENT = get_environment()

        self.VERSION = "0.1.0"
        self.DATABASE_URL = str(os.getenv("SUPABASE_URL"))
        self.DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "t", "yes")



settings = Settings()
import os

from sqlmodel import create_engine
from dotenv import load_dotenv

load_dotenv()

DATABASE = os.environ.get("SUPABASE_PG_URL")
if DATABASE:
    engine = create_engine(str(DATABASE))
    print(engine)
else:
    print("Not connect!")
    print(f"database:{DATABASE}")
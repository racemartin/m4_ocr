import os
import urllib.parse
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    user = os.getenv("DB_USER")
    password = urllib.parse.quote_plus(os.getenv("DB_PASSWORD"))
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    dbname = os.getenv("DB_NAME")
    db_uri = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(db_uri, client_encoding='utf8')

# Instancia global
engine = get_engine()
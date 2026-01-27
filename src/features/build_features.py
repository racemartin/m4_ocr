# -*- coding: utf-8 -*-

import pandas as pd
from sqlalchemy import text
from pathlib import Path
from src.database import engine 

def build_features():
    # 1. Definir rutas
    sql_path = Path(__file__).parent / "create_features_view.sql"
    
    # 2. Leer el SQL
    with open(sql_path, 'r') as f:
        query = f.read()
        
    # 3. Ejecutar en la base de datos
    with engine.begin() as conn:
        conn.execute(text(query))
    
    print("✅ Ingeniería de características completada en la DB.")

if __name__ == "__main__":
    build_features()
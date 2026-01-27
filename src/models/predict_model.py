import pandas as pd
from sqlalchemy import text
from pathlib import Path
# Asumimos que tienes un motor de base de datos configurado en tu src
# Si no, puedes importarlo de donde lo tengas definido
# from src.database import engine 

from src.database import engine

def generate_risk_predictions(conn_engine):
    """
    Crea la vista de riesgos basada en los pesos de importancia 
    calculados previamente mediante SHAP.
    """

    query_view = text("""
    -- 1. Effacer si elle existe
    DROP VIEW IF EXISTS v_top_risques;
    
    -- 2. Creer a nouveau
    CREATE VIEW v_top_risques AS
    WITH scores_risque AS (
        SELECT 
            emp_id,
            age,
            departement,
            poste,
            heures_supplementaires, 
            annees_dans_l_entreprise,
            annee_experience_totale,
            (
                (heures_supplementaires * 3.0) + 
                (annee_experience_totale * 2.01) + 
                ((4 - satisfaction_employee_nature_travail) * 1.51) + 
                -- ("poste_Assistant de Direction" * 1.15) + 
                (fe2_stabilite_manager * 0.95) + 
                ((4 - satisfaction_employee_equilibre_pro_perso) * 0.9) + 
                (age * 0.79) + 
                -- ("statut_marital_Divorcé(e)" * 0.74) + 
                -- ("poste_Représentant Commercial" * 0.26) + 
                (fe7_penibilite_trajet * 0.21)
            ) AS score_risque_composite
        FROM v_features_engineering
        WHERE target_attrition = 0
    )
    SELECT 
        emp_id,
        age,
        departement,
        poste,
        -- La nueva columna que causaba el conflicto
        CASE WHEN heures_supplementaires = 1 THEN 'Oui' ELSE 'Non' END AS heures_supp,
        annees_dans_l_entreprise,
        annee_experience_totale,
        ROUND(CAST(score_risque_composite AS NUMERIC), 2) AS score_risque
    FROM scores_risque
    ORDER BY score_risque_composite DESC
    LIMIT 10;
    """)

    print("🚀 Generando predicciones de riesgo en la base de datos...")
    with conn_engine.begin() as conn:
        conn.execute(query_view)
    print("✅ Vista 'v_top_risques' (Top 10) actualizada con éxito.")

if __name__ == "__main__":
    # Aquí deberías tener tu lógica de conexión al engine
    # Por ahora lo dejamos listo para ser llamado por el Makefile
    from sqlalchemy import create_engine
    # Sustituye con tu URL real o cárgala de variables de entorno
    
    generate_risk_predictions(engine)
    pass
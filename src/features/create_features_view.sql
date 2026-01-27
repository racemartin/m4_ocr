-- create_features_view.sql
DROP VIEW IF EXISTS v_features_engineering CASCADE;

CREATE VIEW v_features_engineering AS
SELECT 
    *,
    
    -- FE1: Ratio de stagnation
    ROUND(
        (CAST(annees_depuis_la_derniere_promotion AS FLOAT) / 
        NULLIF(annees_dans_l_entreprise + 1, 0))::NUMERIC, 
        4
    ) AS fe1_ratio_stagnation,
    
    -- FE2: Stabilité Manager
    ROUND(
        (CAST(annes_sous_responsable_actuel AS FLOAT) / 
        NULLIF(annees_dans_le_poste_actuel + 1, 0))::NUMERIC, 
        4
    ) AS fe2_stabilite_manager,
    
    -- FE3: Indice Job Hopping
    ROUND(
        (CAST(annee_experience_totale AS FLOAT) / 
        NULLIF(nombre_experiences_precedentes + 1, 0))::NUMERIC, 
        4
    ) AS fe3_indice_job_hopping,
    
    -- FE4: Ancienneté relative
    ROUND(
        (CAST(annees_dans_l_entreprise AS FLOAT) / 
        NULLIF(GREATEST(age - 18, 1), 0))::NUMERIC, 
        4
    ) AS fe4_anciennete_relative,
    
    -- FE5: Satisfaction globale
    ROUND(
        ((satisfaction_employee_environnement + 
         satisfaction_employee_nature_travail + 
         satisfaction_employee_equipe + 
         satisfaction_employee_equilibre_pro_perso) / 4.0)::NUMERIC, 
        2
    ) AS fe5_satisfaction_globale,
    
    -- FE6: Risque d'overwork
    ROUND(
        (heures_supplementaires * (1.0 / NULLIF(satisfaction_employee_equilibre_pro_perso + 1, 0)))::NUMERIC, 
        4
    ) AS fe6_risque_overwork,
    
    -- FE7: Pénibilité trajet
    heures_supplementaires * distance_domicile_travail AS fe7_penibilite_trajet,
    
    -- FE8: Valeur de l'expérience
    ROUND(
        (CAST(revenu_mensuel AS FLOAT) / 
        NULLIF(annee_experience_totale + 1, 0))::NUMERIC, 
        2
    ) AS fe8_valeur_experience

FROM v_master_clean;
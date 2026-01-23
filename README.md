
# TechNova Attrition Analysis - Prédisez et comprenez le turnover des employés.

<div align="left">
  <img src="docs/images/logo_technova.png" width="200px" alt="Logo TechNova Partners">
</div>

## **Objectif**: Identifier les causes racines de l'attrition et prédire le départ des collaborateurs à l'aide de **XGBoost** et **SHAP**.

## 🛠️ Technologies

* **Python 3.12+**
* **uv** (Gestionnaire de paquets ultra-rapide)
* **Scikit-learn, XGBoost** (Modélisation)
* **SHAP** (Interprétabilité des modèles)
* **Pandas, Seaborn** (Analyse de données)

## 📦 Installation

```bash
# Cloner le dépôt
git clone https://github.com/racemartin/m4_ocr.git
cd m4_ocr

# Installer les dépendances et créer l'environnement virtuel avec uv
uv sync

# Vérifier l'installation
uv run python -c "import pandas, shap, xgboost; print('Environnement prêt !')"

```

## 🚀 Utilisation

```bash
# Exécuter Jupyter Lab via uv
uv run jupyter lab

# Lancer l'analyse exploratoire automatique
uv run python -m ydata_profiling --config src/data/config_eda.yaml

```

### Notebooks Importants (Ordre d'implémentation)

| Étape | Notebook | Focus | Entrée | Sortie |
| --- | --- | --- | --- | --- |
| **1. EDA** | `01_RC_EDA_Nettoyage.ipynb` | Fusion & Nettoyage | RAW CSV | `interim/data_merged.csv` |
| **2. FE** | `02_RC_Feature_Engineering.ipynb` | Encodage & Features | interim | `processed/data_final.csv` |
| **3. MOD** | `03_RC_Modelisation.ipynb` | Entraînement XGBoost | processed | `models/V1/attrition_v1.joblib` |
| **3. MOD** | `04_RC_Modelisation_V2.ipynb` | Entraînement XGBoost | processed | `models/V2/attrition_v1.pkl` |
| **4. SHAP** | `05_RC_Interpretation_SHAP.ipynb` | **Causes du Turnover** | model | Reports/Figures |

## 📝 Structure du projet

```bash
.
├── .venv               # Environnement virtuel (isolé par uv)
├── data
│   ├── raw             # Fichiers originaux (SIRH, Eval, Sondage)
│   ├── interim         # Données après fusion (.merge())
│   └── processed       # Données prêtes pour XGBoost
├── models              # Modèles sérialisés (.pkl)
├── notebooks           # Expérimentations pas à pas
├── pyproject.toml      # Configuration des dépendances (format uv)
├── src                 # Scripts Python modulaires
└── reports             # Résultats SHAP et graphiques pour Amandine

```

Details 

```bash
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

```


## 📚 Dépendances principales

### Production

* **pandas** : Manipulation des bases RH.
* **xgboost** : Algorithme de boosting pour la prédiction d'attrition.
* **shap** : Calcul des valeurs de Shapley pour expliquer "pourquoi" un employé part.
* **scikit-learn** : Pipelines de transformation et métriques d'évaluation.

### Développement

* **ydata-profiling** : Génération de rapports EDA rapides.
* **yellowbrick** : Visualisation de la performance des classifieurs (Matrice de confusion, ROC).
* **black / flake8** : Garantie d'un code propre et standardisé.

## 👤 Auteur

**Rafael Cerezo Martín**

* Email: [rafael.cerezo.martin@icloud.com](mailto:rafael.cerezo.martin@icloud.com)
* GitHub: [@racemartin](https://github.com/racemartin)

## 📄 Licence

MIT License - voir le fichier [LICENSE](https://www.google.com/search?q=LICENSE) pour plus de détails.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

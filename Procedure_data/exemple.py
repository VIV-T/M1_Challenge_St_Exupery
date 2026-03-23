import os
from google.cloud import bigquery

def query_bigquery_table(project_id: str, dataset_id: str, table_id: str, service_account_key_path: str):
    """
    Exécute un SELECT * sur une table BigQuery et affiche les résultats.

    Args:
        project_id (str): L'ID du projet GCP.
        dataset_id (str): L'ID du dataset BigQuery.
        table_id (str): L'ID de la table BigQuery.
        service_account_key_path (str): Le chemin vers le fichier JSON de la clé du compte de service.
    """
    try:
        # Configure les identifiants du compte de service
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path

        # Initialise le client BigQuery
        client = bigquery.Client(project=project_id)

        # Construit la référence complète de la table
        table_ref = f"`{project_id}.{dataset_id}.{table_id}`"

        # Construit la requête SQL
        query = f"SELECT * FROM {table_ref} LIMIT 10" 

        print(f"Exécution de la requête sur BigQuery: {query}\n")

        query_job = client.query(query)

        # Récupère les résultats
        rows = query_job.result()

        print("Résultats de la requête:")
        for row in rows:
            print(row)

        print(f"\nRequête terminée. {rows.total_rows} lignes récupérées.")

    except Exception as e:
        print(f"Une erreur est survenue: {e}")

# --- Configuration et exécution du script ---
if __name__ == "__main__":
    YOUR_PROJECT_ID = "va-sdh-adl-staging"
    YOUR_DATASET_ID = "aero_insa"
    YOUR_TABLE_ID = "mouvements_aero_insa"
    YOUR_SERVICE_ACCOUNT_KEY_PATH = "config/va-sdh-adl-staging.json"

    query_bigquery_table(
        project_id=YOUR_PROJECT_ID,
        dataset_id=YOUR_DATASET_ID,
        table_id=YOUR_TABLE_ID,
        service_account_key_path=YOUR_SERVICE_ACCOUNT_KEY_PATH
    )
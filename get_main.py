import os
from google.cloud import bigquery
import pandas as pd
import pandas_gbq


# ------------- Global variable -------------

column_list = """
IdMovement,
IdADL,
IdAircraftType,
IdBusinessUnitType,
IdBusContactType,
IdTerminalType,
IdBagStatusDelivery,
NbFlight,
AirportCode,
airlineOACICode,
SysStopover,
AirportOrigin,
AirportPrevious,
ServiceCode,
flightNumber,
OperatorFlightNumber,
FlightNumberNormalized,
OperatorOACICodeNormalized,
LTScheduledDatetime,
LTScheduledTime,
LTExternalDatetime,
LTExternalDate,
LTExternalTime,
Direction,
Terminal,
SysTerminal,
FuelProvider,
ScheduleType,
NbOfSeats,
NbPaxTotal,
etl_origin
"""


additional_condition = """
ORDER BY LTScheduledDatetime DESC
"""


# ------------- Utils -------------

def query_bigquery_table(project_id: str, dataset_id: str, table_id: str, service_account_key_path: str) -> pd.DataFrame:
    """
    Execute a choosen query on our BigQuery table and save the result into a pd.DataFrame and then a csv file.

    Args:
        project_id (str): L'ID du projet GCP.
        dataset_id (str): L'ID du dataset BigQuery.
        table_id (str): L'ID de la table BigQuery.
        service_account_key_path (str): Le chemin vers le fichier JSON de la clé du compte de service.
    """
    try:
        # Configure les identifiants du compte de service
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path

        # Construit la référence complète de la table
        table_ref = f"`{project_id}.{dataset_id}.{table_id}`"

        # Construit la requête SQL
        query = f"SELECT {column_list} FROM {table_ref}  {additional_condition}" 

        ### Way 1 to query the database.        
        # Initialise le client BigQuery
        client = bigquery.Client(project=project_id)
        query_job = client.query(query)
        # Récupère les résultats
        df_res = query_job.to_dataframe(create_bqstorage_client=False)

        ### Way 2 to query the database. 
        # Query execution using "pandas_gbq".
        # print(f"Query execution: \n{query}\n")
        # df_res = pd.DataFrame(pandas_gbq.read_gbq(query, project_id=project_id))

        return df_res

    except Exception as e:
        print(f"Une erreur est survenue: {e}")
        return pd.DataFrame()


# ------------- Script configuration and execution -------------
if __name__ == "__main__":
    YOUR_PROJECT_ID = "va-sdh-adl-staging"
    YOUR_DATASET_ID = "aero_insa"
    YOUR_TABLE_ID = "mouvements_aero_insa"
    YOUR_SERVICE_ACCOUNT_KEY_PATH = "config/va-sdh-adl-staging.json"

    df_res = query_bigquery_table(
        project_id=YOUR_PROJECT_ID,
        dataset_id=YOUR_DATASET_ID,
        table_id=YOUR_TABLE_ID,
        service_account_key_path=YOUR_SERVICE_ACCOUNT_KEY_PATH
    )

    print(df_res)
    df_res.to_csv("data/main.csv", encoding='utf-8', index=False)
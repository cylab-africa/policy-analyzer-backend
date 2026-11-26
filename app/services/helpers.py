from app.db.base_db import mongo_client
from pymongo.errors import OperationFailure
import pycountry
import json


def write_country_law_to_db(file_path_country: str):

    try:
        with open(file_path_country, 'r', encoding='utf-8') as file:
            data = file.read()
        laws = json.loads(data)
        mongo_client.get_collection('policy_law_docs').insert_many(laws)
        print(
            f"Successfully written {len(laws)} country laws to the database.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading file {file_path_country}: {str(e)}")
    except (ConnectionError, OperationFailure) as e:
        print(f"Database error: {str(e)}")


def write_digital_id_data(file_path_digital_id: str):
    try:
        with open(file_path_digital_id, 'r', encoding='utf-8') as file:
            data = file.read()

        mongo_client.get_collection('digital_ids_docs').insert_many(json.loads(data))
        print(
            f"Successfully written {len(data)} digital ID data to the database.")

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading file {file_path_digital_id}: {str(e)}")
    except OperationFailure as e:
        print(f"Database error: {str(e)}")

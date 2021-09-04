import os
from time import time
import sys
from bson.objectid import ObjectId
from celery import Celery
from pymongo import MongoClient

# libraries for Competency maps
from pathlib import Path
from competency_maps.competency_maps import CompetencyMap
from competency_maps.utils.map_config import CompetencyMapConfig

# App modules
from utils.helpers import read_data, get_resources, write_to_database, clear_database, get_name_levels
from configuration.sys_config import config
import configuration.celery_config as celery_config

celery_app = Celery("worker")
celery_app.config_from_object(celery_config)

celery_app.conf.task_routes = {
    "worker.celery_app.create_map2": "task-queue"}

celery_app.conf.update(task_track_started=True)
dbname = config.database_name

@celery_app.task(acks_late=True)
def create_map2(map_id: str) -> str:
    '''
        The methd for creating the competency map. It is called in the API call method ,
        It uses the functions from helpers clear_database, get_resources, write_to_database for communicating with the Mongodb.
        Prints "Done competency map"
        args:
            map_id : str the id of the competency map
        return : 
            void
    '''
    client = MongoClient()
    db = client[dbname]

    map_id_string = map_id
    map_id = ObjectId(map_id)

    output_folder = config.map_output
    _, levels = get_name_levels(map_id, db)

    map_config = CompetencyMapConfig()
    map_config.NUM_LEVELS = levels

    # Clear database
    clear_database(map_id, db)
    # Fetch resources
    resources = get_resources(map_id, db)
    # Compentecy map creation
    test = CompetencyMap(map_id_string, map_config, resources, output_folder)
    topics, resources, resource_topic_mapping, map_summary = test.create_map()
    # From helpers
    write_to_database(map_id, db, map_config, topics, resources, resource_topic_mapping)

    client.close()

    print("Done creating competency Mapping")

    return 1


if __name__ == "__main__":
    pass

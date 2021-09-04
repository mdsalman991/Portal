from api_gateway.worker.celery_app import celery_app

#libraries for Competency maps
from pathlib import Path
from competency_maps.competency_maps import CompetencyMap
from competency_maps.utils.map_config import CompetencyMapConfig
import os
#App modules
from api_gateway.utils.helpers import read_data, get_resources, write_to_database, clear_database, get_name_levels
from api_gateway.configuration.sys_config import config


@celery_app.task(acks_late=True)
def create_map(map_id: str) -> str:
    '''
        The methd for creating the competency map. It is called in the API call method
        It uses the functions from helpers clear_database, get_resources, write_to_database for communicating with the Mongodb.
        Prints "Done competency map"
        args:
            map_id : str the id of the competency map
        return : 
            void
    '''
    output_folder = config.map_output
    name, levels = get_name_levels(map_id)

    map_name = name

    map_config = CompetencyMapConfig()

    
    # map_config.NUM_LEVELS = 10 # get from database
    map_config.NUM_LEVELS = levels
    
    # Clear database
    clear_database(map_id)
    # Fetch resources
    resources = get_resources(map_id)

    #Compentecy map creation
    test = CompetencyMap(map_name, map_config, resources, output_folder)
    topics, resources, resource_topic_mapping, map_summary = test.create_map()
    #From helpers
    write_to_database(map_id,map_config,topics,resources,resource_topic_mapping)

    print("Done creating competency Mapping")
    
    return 1


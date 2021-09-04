
'''
The functions which are used by the create_map API.
Mostly invlolves communication with the MongoDB and reading files. 


Methods : 
    read_data()
    clear_database()
    write_to_database()

'''

from configuration.sys_config import config


import pandas as pd
import time
from datetime import datetime
import asyncio
import os
from pathlib import Path
from bson.objectid import ObjectId


# dbname = config.databasename
cmaps = config.cmaps
cmap_resources = config.cmap_resources
cmap_resource_topic = config.cmap_resource_topic
cmap_topics = config.cmap_topics
resources = config.resources
lmaps = config.lmaps
lmap_resources = config.lmap_resources

def _read_from_txt(path):
    f = open(path,'r')
    ans = ""
    for line in f:
        ans += line
    return ans

def read_data(resources):
    '''
        return a dataframe as below
        ID | Data
        uses a function read_from_txt can be changed to read video etc.
    
        args: 
            resources : a dictionary with key as the resource_id and the value of the key as the resources location
        return:
            returns a dataframe as below 
            |"resource_id" | "description" |
            | 12           | .......
             ....
             ....  
    '''
    
    frame = []
    print(resources.keys())
    for i in resources.keys():
        data = _read_from_txt(resources[i])
        frame.append((i,data))
    data_frame = pd.DataFrame(columns=["resource_id","description"],data=frame)
    return data_frame

def clear_database(map_id, db):
    '''
        This is the function that flushed the database so that it can be updated after competency map creation.
        Connects to the mongoDB and then deletes the entries in tables CMAP_RESOURCE_TOPIC_TABLE and CMAP_TOPICS which have entries with the given map_id
        args : 
            map_id : int that gives the map id
        reutrn: 
            void
    '''
    db[cmap_resource_topic].delete_many({'mapId':map_id})
    db[cmap_topics].delete_many({'mapId':map_id})
    

def get_name_levels(map_id, db):
    rows = db[cmaps].find_one({'_id':map_id})
    name = rows['mapName'] # SEE IF THIS SHOUDL BE CHANGED
    levels = rows['numLevels'] # HERE 
    return name,levels

def get_resources(map_id, db):
    '''
        This is the method that access the database to get the resources locations using the map_id and then uses the read_data to get the dataframe
        args : 
            map_id : int the map id that is used to get the resources locations using the resources ids of all the rows having the map id as given here.
        return : 
            a dataframe of resources. same as the return of read_data
    
    '''

    resource_ids = db[cmap_resources].find({'mapId':map_id})
    ans = {}                                                        
    for res in resource_ids:
        print(res)
        res_id = res['resourceId']
        res_repo = db[resources].find_one({'_id':res_id})  #CHANGE THE NAME BASED ON THE DATABASE    
        ans[res_id] = res_repo['resourceLocation']['convertedLocation'] + ".txt"
    return read_data(ans)




def write_to_database(map_id, db, map_config,topics,resources,resource_topic_mapping):
    '''
        This function writes and updates to the database after the creation of the competency map.
        Genrally used for updating the results of the map creation to the MongoDB.
        args:
            map_id : int the map id of the created map
            map_config : the configuration of the map created
            topics : Topics dataframe of the output 
            resources : Resources dataframe
            resource_topic_mapping : resource_topic_mapping dataframe of the output.
            Last three arguments are the outputs of the .create_map() method . 
        We can use the clean_database method to flush the entries before writing.
    '''

    required_cols = ['resource_id', 'resource_volume', 'topic_name', 'topic_type', 'topic_volume',
                     'document_mapped_probability']
    
    resource_topic_mapping = resource_topic_mapping[required_cols].drop_duplicates()

    #adding map_id column for required dataframes
    resource_topic_mapping['map_id'] = map_id
    topics['map_id'] = map_id
    resources['map_id'] = map_id
    
    
    #Converting 
    rtm = resource_topic_mapping.to_dict(orient='records')
    r = resources.to_dict(orient='records') # we have to update
    t = topics.to_dict(orient='records')

    #changing attribute names
    #topics
    for i in t:
        i["mapId"] = i.pop("map_id")
        i["topicName"] = i.pop("topic_name")
        i["locationX"] = i.pop("X")
        i["topicClusterId"] = i.pop("topic_cluster_id")
        i["topicClusterProbability"] = i.pop("topic_cluster_probability")
        i["topicType"] = i.pop("topic_type")

    #rt mapping
    for i in rtm:
        i["mapId"] = i.pop("map_id")
        i["resourceId"] = i.pop("resource_id")
        i["topicName"] = i.pop("topic_name")
        i["resourceMappedProbability"] = i.pop("document_mapped_probability")

    db[cmap_resource_topic].insert_many(rtm)
    lmap_id = db[lmaps].find_one({"cMapId":map_id})["_id"]

    for res in r:
        db[cmap_resources].update_one({'mapId':map_id,'resourceId':res['resource_id']},{'$set':{'locationX':res['X'],'locationY':res['Y'],'normX':res['norm_X'],'normY':res['norm_Y']}})
        db[lmap_resources].insert_one({'mapId':lmap_id,"presentInCompetency":True, 'resourceId':res['resource_id'],'locationX':res['X'],'locationY':res['Y'],'normX':res['norm_X'],'normY':res['norm_Y']})

    db[cmap_topics].insert_many(t)

    db[cmaps].update_one({'mapName':map_id},{'$set':{'numTopics':len(t),'updatedOn':datetime.now()}})


import os


resources = os.listdir("/home/srihari/iiitb/sem_8/wsl/Navigated-Learning/learning_maps/nlp_maps/data/course_tmp")
# resources = os.listdir("../sample/")


entries = []
res_t = []
for i in range(len(resources)):
    entries.append({'resourceName':i+1,'resourceLocation':{"convertedLocation":"/home/srihari/iiitb/sem_8/wsl/Navigated-Learning/learning_maps/nlp_maps/data/course_tmp/{}".format(resources[i])}})
    res_t.append({'mapId':1,'resourceId':i+1,'locationX':0,'locationY':0,'normX':0,'normY':0})

print(entries,res_t)
from pymongo import MongoClient
client = MongoClient()
db = client['wsl_db']
# db.cmap_resources.insert_many(res_t)
db.resources.insert_many(entries)

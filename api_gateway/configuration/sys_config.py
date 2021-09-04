from competency_maps import competency_maps
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_name = "portal"
    map_output = "./output/"
    cmaps = 'competencymaps'
    cmap_topics = 'cmaptopics'
    resources = 'resources'
    cmap_resources = 'cmapresources'
    cmap_resource_topic = 'cmapresourcetopics'
    lmaps = "learningmaps"
    lmap_resources = "learningmapresources"
config = Settings()

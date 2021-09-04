import curlify
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/generate_trailer")
async def generate(template: int=2, voiceover: int=1, subtitle: int=0):

    if template==1:
        data = {"response": "Implementation is Pending"}
    elif template==2:
        data = {"response":{"view_area":{"trailer_path":"data/generated/trailer/trailer_template_2.mp4","description_1":"DEMO: This trailer is about Tagging problems in HMMs","description_2":"DEMO: This trailer is made from user selected 10 resources from Pathway #10."},"frame_editing_area":[{"frame_number":1,"theme":"title","title":"A New Approach to the Tagging Problem Based on Hidden Markov Models","start_time":0,"duration_time":4,"frame_path":"data/generated/frames/1.png","color_text":"true","editables":["title","duration_time","color_text"]},{"frame_number":2,"theme":"author","name":"Prof. Jake Mathew","affiliation":"from Purdue University","name_image":"data/default/author/","affiliation_image":"data/default/author/","name_image_update_path":"data/user/author/","affiliation_image_update_path":"data/user/author/","start_time":4,"duration_time":6,"frame_path":"data/generated/frames/2.png","color_text":"true","editables":["name","name_image","affiliation","affiliation_image","duration_time","color_text"]},{"frame_number":3,"theme":"outline","outline_1":"A Brief Overview of Markov Model Tagging","outline_2":"Generative Model for Supervised Learning","outline_3":"Hidden Markov Models","outline_4":"The viterbi algorithm for highest scoring tag sequences","start_time":10,"duration_time":9,"frame_path":"data/generated/frames/3.png","bold_text":"true","editables":["outline_1","outline_2","outline_3","outline_4","duration_time","bold_text"]},{"frame_number":4,"theme":"statistics","statistics_1":"1 hr of total readtime","statistics_2":"5 resources in Total","statistics_3":"Videos/Text resources","statistics_4":"Active Discussion Forum","statistics_5":"Assignments included","start_time":19,"duration_time":6,"frame_path":"data/generated/frames/4.png","color_text":"true","editables":["statistics_1","statistics_2","statistics_3","statistics_4","statistics_5","color_text"]},{"frame_number":5,"theme":"validation","data_1":"800 users with completion rate of 92%","start_time":25,"duration_time":8,"frame_path":"data/generated/frames/5.png","color_text":"true","editables":["data_1","duration_time","color_text"]},{"frame_number":6,"theme":"cta","data_1":"Sounds Interesting?","data_2":"Start your journey","start_time":33,"duration_time":4,"frame_path":"data/generated/frames/6.png","editables":["data_1","data_2","duration_time"]}],"meta_editing_area":{"subtitle_checkbox":"true","subtitle_path_download":"data/generated/subtitle","subtitle_path_upload":"data/user/subtitle","wordcloud_checkbox":"true","wordcloud_path_download":"data/generated/wordcloud","wordcloud_path_upload":"data/user/wordcloud","voiceover_checkbox":"true","voiceover_path_male_1":"data/default/voices/male_1.mp4","voiceover_path_male_2":"data/default/voices/male_2.mp4","voiceover_path_female_1":"data/default/voices/female_1.mp4","voiceover_path_female_2":"data/default/voices/female_2.mp4","editables":["subtitle_checkbox","wordcloud_checkbox","voiceover_checkbox"]},"suggestions_area":{"titles":["title_1","title_2"],"definitions":["def_1","def_2"]}}}
    elif template==3:
        data = {"response": "Implementation is Pending"}

    json_compatible_item_data = jsonable_encoder(data)
    return JSONResponse(content=json_compatible_item_data)


@app.get("/title_generation")
async def title_generation():
    return

@app.get("/tts_engine")
async def tts():
    return
    
    # curl -X POST -u "apikey:5TPRkfXQe6b3AO_Jp0ZK2b4uxKNmy-ZomDZB5ZfKjUhJ" \
    # --header "Content-Type: application/json" \
    # --header "Accept: audio/wav" \
    # --data "{\"text\":\"Hi, my name is olivia. This is how i speak, please note the fluency, rate, pronunciation and texture before making your choice. Thank you\"}" \
    # --output Pictures/template_2/female_1.wav \
    # "https://api.eu-gb.text-to-speech.watson.cloud.ibm.com?voice=en-US_Olivia"
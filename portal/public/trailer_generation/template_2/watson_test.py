import json
from os.path import join, dirname
from ibm_watson import TextToSpeechV1
from ibm_watson.websocket import SynthesizeCallback
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator = IAMAuthenticator('5TPRkfXQe6b3AO_Jp0ZK2b4uxKNmy-ZomDZB5ZfKjUhJ')
service = TextToSpeechV1(authenticator=authenticator)
service.set_service_url('https://api.eu-gb.text-to-speech.watson.cloud.ibm.com')

voices = service.list_voices().get_result()
print(json.dumps(voices, indent=2))

with open(join(dirname(__file__), 'female_2.wav'),
          'wb') as audio_file:
    response = service.synthesize(
        'Hi, my name is lisa. This is how i speak, please note the fluency, rate, pronunciation and texture before making your choice. Thank you!', accept='audio/wav',
        voice="en-US_LisaV3Voice").get_result()
    audio_file.write(response.content)
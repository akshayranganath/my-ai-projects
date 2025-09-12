from marvin.beta.assistants import Assistant
import requests

# define a custom tool function 
def visit_url(url:str)->str:
    """Fetch contents of a URL"""
    return requests.get(url).content.decode()

ai = Assistant(tools=[visit_url])
ai.say("Can you tell me the current temperature in Livermore, California on weather.com?")
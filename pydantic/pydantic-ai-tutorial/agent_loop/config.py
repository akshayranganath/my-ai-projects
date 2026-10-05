# Load .env, build the model (MODEL_NAME, BASE_URL, API_KEY), resolve the workspace (SAFE_FILE_SYSTEM_FOLDER)
from dotenv import load_dotenv
import os
from pathlib import Path

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider


# open telemetry for tracing
import logfire

# get the environment variables
load_dotenv()

# initialize logging
logfire.configure()
logfire.instrument_pydantic_ai()

# build the model
model = OpenAIChatModel(
    model_name=os.environ['MODEL_NAME'],
    provider=OpenAIProvider(
        base_url = os.environ['BASE_URL'] ,
        api_key= os.environ['API_KEY']
    ),
    profile=OpenAIModelProfile(supports_json_object_output=False),
)

# build the safe path for workspace and code execution
# get the config file's folder
CONFIG_DIR = Path(__file__).resolve().parent

WORKSPACE = os.environ['SAFE_FILE_SYSTEM_FOLDER']
WORKSPACE = Path(WORKSPACE).expanduser()
WORKSPACE = (CONFIG_DIR / WORKSPACE).resolve()

# auto create folder if needed
WORKSPACE.mkdir(parents=True, exist_ok=True)
    
def get_llm()->OpenAIChatModel:
    return model

def get_workspace()->Path:
    return WORKSPACE

def _agent_debug_log(flow:str, source:str, message:str, context:dict)->None:
    logfire.debug(f"Agent Debug Log: {flow} {source} {message} {context}")
    print(f"Agent Debug Log: {flow} {source} {message} {context}")


if __name__=="__main__":
    print(get_llm())    
    print(get_workspace())
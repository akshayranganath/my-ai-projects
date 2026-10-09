import os
from dotenv import load_dotenv
load_dotenv()

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
#from openai import AsyncOpenAI

model = OpenAIChatModel(
    model_name=os.environ.get('MODEL_NAME'),
    provider=OpenAIProvider(
        base_url = os.environ.get('BASE_URL'),
        api_key= os.environ.get('API_KEY')
    )
)

agent = Agent(
    model,
    instructions="You are a friendly pirate. Keep your responses friendly but in pirate language."
)

result = agent.run_sync("Say hello and explain you are running locally")
print(result.output)
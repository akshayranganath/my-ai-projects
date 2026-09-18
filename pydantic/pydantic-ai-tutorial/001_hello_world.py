from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
#from openai import AsyncOpenAI

model = OpenAIChatModel(
    model_name="nvidia/nemotron-3-nano-4b",
    provider=OpenAIProvider(
        base_url = 'http://localhost:1234/v1',
        api_key='lm-studio'
    )
)

agent = Agent(
    model,
    instructions="You are a friendly pirate. Keep your responses friendly but in pirate language."
)

result = agent.run_sync("Say hello and explain you are running locally")
print(result.output)

import traceback
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

os.environ['PYDANTIC_AI_NO_BANNER'] = "1"


model = OpenAIChatModel(
    model_name=os.environ.get('MODEL_NAME'),
    provider=OpenAIProvider(
        base_url = os.environ.get('BASE_URL'),
        api_key= os.environ.get('API_KEY')
    )
)

agent = Agent(
    model,
    instructions="You are a pompous language expert. You use crazy hard English words in even simple conversations."
)

async def main()->None:

    message_history = []
    print("Local assisstant started. Type 'exit' to quit.")

    while True:
        try:            
            user_input = await asyncio.to_thread(
                input,
                "You: "
            )
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        user_input = user_input.strip()

        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        
        try:
            result = await agent.run(
                user_input,
                message_history=message_history
            )

            print(f"Assisstant: {result.output}")

            # preserve history
            message_history = result.all_messages()
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
import errno
import os
from dotenv import load_dotenv
load_dotenv()

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import traceback

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

def main()->None:

    message_history = []

    print("Local assisstant started. Type 'exit' to quit.")


    while True:

        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        
        try:
            result = agent.run_sync(
                user_input,
                message_history=message_history
            )

            print(f"Assisstant: {result.output}")

            # preserve history
            message_history = result.all_messages()
        except Exception as e:
            print(f"Error: {errno}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
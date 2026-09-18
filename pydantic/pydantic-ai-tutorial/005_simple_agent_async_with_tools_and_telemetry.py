
import traceback
import asyncio
import os
#from pprint import pprint

from pydantic_ai import Agent, capabilities
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.capabilities import WebSearch

# open telemetry for tracing
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

model = OpenAIChatModel(
    model_name="nvidia/nemotron-3-nano-4b",
    provider=OpenAIProvider(
        base_url = 'http://localhost:1234/v1',
        api_key='lm-studio'
    )
)

agent = Agent(
    model,
    instructions="""You are a helpful assistant. Use web search when the user asks about current information
    or explicitly asks you to search. Use it no more than twice for one user request.
    After receiving useful search results, answer the user instead of searching again.When using search, always include the source URLs in the answer.""",
    capabilities=[
        WebSearch(local="duckduckgo")
    ]
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
            
            # debug details
            #print("\nExecution messages: ")
            #pprint(result.new_messages())
            
            # preserve history
            message_history = result.all_messages()
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
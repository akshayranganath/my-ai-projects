"""
Agent routing with Jev.

Instead of one general-purpose agent, this example defines three specialised
agents (elementary, research, general). For every user prompt, the Jev
(typesafe_sdk) client classifies the question first, and the prompt is then
routed to the agent best suited to answer it.
"""

# Jev SDK: `TypeSafeClient` talks to the Jev service and `Choice` describes a
# multiple-choice question whose answer must be one of the given criteria keys.
from typesafe_sdk import Choice, TypeSafeClient

import traceback
import asyncio
import os
import shlex
from pathlib import Path
from dotenv import load_dotenv
# load environment variables (model settings, Jev / Logfire credentials) from .env
load_dotenv()


from pydantic_ai import Agent, RunContext, capabilities
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.capabilities import WebSearch

# open telemetry for tracing
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

# OpenAI-compatible model configured via MODEL_NAME, BASE_URL and API_KEY in .env
model = OpenAIChatModel(
    model_name=os.environ.get('MODEL_NAME'),
    provider=OpenAIProvider(
        base_url = os.environ.get('BASE_URL'),
        api_key= os.environ.get('API_KEY')
    )
)

# let's create 3 agents
# All three share the same model and web search capability; only the
# instructions differ, which changes the tone and depth of the answers.

# 1. research agent: detailed, technical answers with citations
deep_research_agent = Agent(
    model,
    instructions="""
    You are an agent that can perform deep research. Your audience is technically oriented PhD candidates. Include foundational details.
    If needed, include subject specific technical terms. Don't be afraid to use the references and ensure you follow the citation
    methodology that is generally adopted by research students.

    Use web search when the user asks about current information
    or explicitly asks you to search. Use it no more than twice for one user request.
    After receiving useful search results, answer the user instead of searching again.When using search, always include the source URLs in the answer.
    """,
    capabilities=[
        # DuckDuckGo search runs locally, so no search API key is needed
        WebSearch(local="duckduckgo")
    ]
)

# 2. elementary agent: short, simple, jargon-free answers for children
elementary_answer_agent = Agent(
    model,
    instructions="""
    You are a friendly agent tasked to answer elementary school questions. Keep sentences short and simple. Don't include jargon.
    Use simple examples. Don't include heavy references.

    Use web search when the user asks about current information
    or explicitly asks you to search. Use it no more than twice for one user request.
    After receiving useful search results, answer the user instead of searching again.
    """,
    capabilities=[
        WebSearch(local="duckduckgo")
    ]
)

# 3. general agent: everyday assistant, also used as the fallback
general_answer_agent = Agent(
    model,
    instructions="""
    You are a helpful assistant. Use web search when the user asks about current information
    or explicitly asks you to search. When using search, always include the source URLs in the answer.

    Use web search when the user asks about current information
    or explicitly asks you to search. Use it no more than twice for one user request.
    After receiving useful search results, answer the user instead of searching again.When using search, always include the source URLs in the answer.
    """,
    capabilities=[
        WebSearch(local="duckduckgo")
    ]
)

# maps the label returned by the classifier to the agent that handles it.
# keys must match the `criteria` keys used in `identify_agent_type`.
agent_type_mapping = {
    "elementary": elementary_answer_agent,
    "research": deep_research_agent,
    "general": general_answer_agent
}

# initialize jev client
jev_client = TypeSafeClient()

def identify_agent_type(prompt:str)->str:
    """Classify the prompt as 'elementary', 'research' or 'general' using Jev.

    Returns None if classification fails, which makes the caller fall back to
    the general agent.
    """
    try:
        # `system_one` evaluates the prompt (`state`) against each question.
        # Here there is a single multiple-choice question, `query_type`, and
        # the answer is constrained to one of the `criteria` keys.
        response = jev_client.system_one(
            state = prompt,
            questions = {
                "query_type": Choice(
                    instructions = "What kind of a prompt/question is this?",
                    criteria={
                        "elementary": "Simple question suitable for elementary children",
                        "research": "A deep research problem that is suitable for advanced students",
                        "general": "Suitable for any audience but, not necessarily suitable for elementary children"
                    }
                )
            }
        )        
        # answers are keyed by question name; `.choice` is the selected label
        return response.answers.get('query_type').choice
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

## now, let us run this in a loop

async def main()->None:

    # conversation so far, shared across all agents so context carries over
    # even when consecutive prompts are routed to different agents
    message_history = []
    print("Local assisstant started. Type 'exit' to quit.")

    while True:
        try:            
            # run the blocking `input()` in a thread so the event loop isn't blocked
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

        # ignore empty lines
        if not user_input:
            continue    
        
        try:
            # step 1: classify the prompt with Jev
            agent_type = identify_agent_type(user_input)
            # step 2: pick the matching agent; unknown or failed classification
            # falls back to the general agent
            selected_agent = agent_type_mapping.get(agent_type, general_answer_agent)
            # step 3: let the selected agent answer, with the prior conversation
            result = await selected_agent.run(
                user_input,
                message_history=message_history,
            )
            print(f"Assisstant-{agent_type}: {result.output}")

            # preserve history
            message_history = result.all_messages()
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

if __name__=="__main__":
    asyncio.run(main())

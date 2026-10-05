import asyncio
import traceback

from anthropic.types import Usage
from websockets.http11 import USER_AGENT

from pydantic_ai.usage import UsageLimits

from config import WORKSPACE, _agent_debug_log
from deps import AgentDeps
from agent import agent, run_goal


async def main()->None:

    deps = AgentDeps(workspace=WORKSPACE)
    message_history = []
    print("Agent ready. Type `exit` to quit.")

    async with agent:
        while True:
            try:
                user_input = (
                    await asyncio.to_thread(input, "You: ")
                ).strip()
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.lower() == "exit":
                break
            if not user_input:
                continue            

            try:
                # #region agent log
                _agent_debug_log(
                    "B",
                    "main.py:main",
                    "user turn",
                    {"is_goal": user_input.startswith("goal:"), "prompt_prefix": user_input[:24]},
                )
                # #endregion
                if user_input.startswith("goal:"):
                    await run_goal(user_input.removeprefix("goal:").strip(), deps)
                    continue
                else:
                    result = await agent.run(
                        user_input,
                        message_history=message_history,
                        deps=deps,
                        usage_limits=UsageLimits(request_limit=15)
                    )
                    print(f"Agent: {result.output}")
                    message_history = result.all_messages()
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
        print("Goodbye!")


if __name__=="__main__":
    asyncio.run(main())
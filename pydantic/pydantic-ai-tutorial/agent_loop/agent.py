from pydantic_ai import Agent, UsageLimits, RunContext, PromptedOutput
from pydantic import BaseModel, Field

from deps import AgentDeps
from config import model, _agent_debug_log
from file_access import file_tools
from search import web_search, SEARCH_INSTRUCTIONS
from code_runner import code_tools, CODE_INSTRUCTIONS


agent = Agent(
    model,
    deps_type = AgentDeps,
    toolsets = [file_tools, code_tools],
    capabilities=[web_search],
    retries=3,
    instructions="You are a helpful assistant. Use the file tools when necessary to answer questions.\n"\
        + SEARCH_INSTRUCTIONS + "\n" + CODE_INSTRUCTIONS    
)

@agent.instructions
def workspace_hint(ctx: RunContext[AgentDeps]) -> str:
    return (
        f"The workspace folder is {ctx.deps.workspace}. "
        "Always pass absolute paths inside this folder to the fs_ tools. "
        f"Code run with run_python executes in {ctx.deps.workspace / 'scratch'}."
    )

class StepReport(BaseModel):
    done : bool = Field(description='True only when the whole goal is complete and verified.')
    summary : str = Field(description='What you did in this step.')
    next_step: str | None = Field(default=None, description="What you will do next if not done.")

MAX_STEPS = 5

async def run_goal(goal:str, deps: AgentDeps)->None:
    history = []
    prompt = goal

    for step in range(1, MAX_STEPS + 1):
        # #region agent log
        _agent_debug_log(
            "A",
            "agent.py:run_goal",
            "goal step uses PromptedOutput",
            {"step": step, "output_type": "PromptedOutput(StepReport)"},
        )
        # #endregion
        result = await agent.run(
            prompt,
            message_history = history,
            deps = deps,
            output_type = PromptedOutput(StepReport),
            usage_limits = UsageLimits(request_limit=15)
        )

        report = result.output
        history = result.all_messages()
        print(f"[step {step}] {report.summary}")
        if report.done:
            print("Goal complete.")
            return
        prompt = f"Continue toward the goal. Your planned next step: {report.next_step}"
    print(f"Stopped after {MAX_STEPS} steps without finishing.")

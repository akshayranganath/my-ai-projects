
import traceback
import asyncio
import os
import shlex
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


from pydantic_ai import Agent, RunContext, capabilities
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.capabilities import WebSearch

# open telemetry for tracing
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

model = OpenAIChatModel(
    model_name=os.environ.get('MODEL_NAME'),
    provider=OpenAIProvider(
        base_url = os.environ.get('BASE_URL'),
        api_key= os.environ.get('API_KEY')
    )
)

# first create a safe folder for access
workspace = os.environ.get('SAFE_FILE_SYSTEM_FOLDER')

agent = Agent(
    model,
    instructions="""You are a helpful assistant. Use web search when the user asks about current information
    or explicitly asks you to search. Use it no more than twice for one user request.
    After receiving useful search results, answer the user instead of searching again. When using search, always include the source URLs in the answer.            
    
    Use file system MCP when trying to write, edit or read from a file. If the file has an extension `.md`, user expects a markdown file. So format it accordingly.
    You can also use a limited set of shell commands using a tool provided. Use it to search or list files. """,    
    capabilities=[
        WebSearch(local="duckduckgo")
    ]    
)

# let's add a tool for bash commands
ALLOWED_COMMANDS = {
    "pwd",
    "ls",
    "cat",
    "find",
    "grep",
    "head",
    "tail",
    "git"
}

@agent.tool_plain
async def run_workspace_command(command: str)->str:
    """ Run limited read oriented command in the agent workspace"""
    parts = shlex.split(command)

    if not parts:
        return "No command provided"
    
    program = parts[0]

    if program not in ALLOWED_COMMANDS:
        raise ValueError(f"Command not allowed: {program}")
    
    # Block absolute value path and parent directory traversal
    for argument in parts[1:]:
        path_argument = Path(argument)
        if ( path_argument.is_absolute() or ".." in path_argument.parts ):
           raise ValueError( "Only workspace-relative paths are allowed") 
    
    process = await asyncio.create_subprocess_exec(
        *parts,
        cwd=workspace,
        env={"PATH": "/usr/bin:/bin"},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=20,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        return "Command timed out"
    
    output = stdout.decode(errors="replace")
    error = stderr.decode(errors="replace")

    return (
        f"Exit code: {process.returncode}\n"
        f"STDOUT: {output}\n"
        f"STDERR: {error}"
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
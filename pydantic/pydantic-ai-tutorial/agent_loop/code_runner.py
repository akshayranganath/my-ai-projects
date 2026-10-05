import asyncio
import sys

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from deps import AgentDeps

TIMEOUT_SECONDS = 15
MAX_OUTPUT_CHARS = 2000

code_tools = FunctionToolset()


CODE_INSTRUCTIONS = """
Use run_python when the user wants code executed or tested. Print results to stdout.
To test code, add assert checks under `if __name__ == "__main__":`.
If a run fails, read the error, fix the code, and run it again (at most 3 times).
If the user only wants to see the code, show it and do not run it.
"""

def _strip_code_fence(code: str)->str:
    """Drop ```python fence if the model wrapped the source in one."""
    text = code.strip()

    if not text.startswith("```"):
        return text

    lines = text.splitlines()[1:] # removes the first ```python line
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    
    return "\n".join(lines).strip()

def _truncate(text: str) -> str:
    """Keep the tail of long output; that is where the trackback ends."""
    text = text.strip()
    return text if len(text) <= MAX_OUTPUT_CHARS else text[-MAX_OUTPUT_CHARS:]

def _format_output(stdout: bytes, stderr: bytes, returncode: int|None, timed_out:bool)->str:

    parts = [f"Timed out after {TIMEOUT_SECONDS} seconds" if timed_out else f"Exit code: {returncode}"]
    out = stdout.decode(errors="replace").strip()
    err = stderr.decode(errors="replace").strip()

    if out:
        parts.append(f"STDOUT:\n{out}")
    if err:
        parts.append(f"STDERR:\n{err}")
    return _truncate("\n\n".join(parts))

@code_tools.tool
async def run_python(ctx: RunContext[AgentDeps], code:str) -> str:
    """Run a complete Python script and return it's exit code, stdout and stderr"""

    # first write the code to a temporary file
    scratch = ctx.deps.workspace/"scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    script = scratch / "snippet.py"
    script.write_text(_strip_code_fence(code) + "\n", encoding='utf-8')

    # run a sub-process to execute the code.
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        cwd=scratch,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(),timeout=TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        process.kill()
        stdout, stderr = await process.communicate()
        return _format_output(
            stdout,
            stderr,
            process.returncode,
            timed_out=True
        )
    return _format_output(
            stdout,
            stderr,
            process.returncode,
            timed_out=False
        )
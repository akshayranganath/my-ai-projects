from fastmcp.client.transports import StdioTransport
from pydantic_ai.mcp import MCPToolset

from config import WORKSPACE


# control access to safe tools
# refer to https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem for the exact names
FS_TOOLS = [
    "read_text_file",
    "write_file",
    "edit_file",
    "list_directory",
    "search_files",
    "get_file_info"
]


file_tools = (
        MCPToolset(
        StdioTransport(
            command = "npx",
            args = [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                str(WORKSPACE)
            ]
        )    
    )
    .filtered(lambda ctx, tool_def: tool_def.name in FS_TOOLS)
    .prefixed('fs')
)


 
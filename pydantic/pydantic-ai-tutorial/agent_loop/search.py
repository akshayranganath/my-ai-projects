from pydantic_ai.capabilities import WebSearch

web_search = WebSearch(local="duckduckgo")

SEARCH_INSTRUCTIONS = """
Use web search only when the user asks about current information or explicitly asks you to search the web.
Search no more than twice per user request. After useful results arrive, answer instead of searching again.
Always include the source URLs in your answer.
"""
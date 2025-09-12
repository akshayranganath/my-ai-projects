import boto3
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.providers.bedrock import BedrockProvider
import marvin
from marvin import Agent, Task


session = boto3.session.Session(profile_name='aws_sol')
    
bedrock_client = session.client('bedrock-runtime' )

bedrock_model = BedrockConverseModel(    
    model_name="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    provider=BedrockProvider(bedrock_client=bedrock_client)
)

openai_model = "openai:gpt-5-2025-08-07"

# in this code, we'll use Bedrock for creation and research while OpenAI for editing

researcher = Agent(
    model=bedrock_model,
    name="researcher",
    description="A researcher who can search the web for information",
    instructions="Perform a thorough research on the given topic and return the results in a structured format. If required, you can use multiple sources to gather information."    
)

writer = Agent(
    model=bedrock_model,
    name="writer",
    description="A writer who can write a blog post",
    instructions="Write an article suitable for moderately technical audience. Use analogies when the technology is hard to explain. The output should be in markdown format. Don't use more than 3 analogies. Keep the blog article less than 1000 words."
)

editor = Agent(
    model=openai_model,
    name="editor",
    description="An editor who can edit a blog post",
    instructions="Edit the draft to make it suitable for the target audience. Correct any grammatical errors and improve the flow of the article. Ensure the output is in markdown format."
)


blog_draft = None
with marvin.Thread() as thread:
    research = researcher.run("Analyze the current state of fusion research and the feasibility of using it for generating commercial electricity.")
    draft = writer.run("Write an article suitable for moderately technical audience. Use analogies when the technology is hard to explain. The output should be in markdown format.", context={"research": research})
    edit = editor.run("Edit the draft to make it suitable for the target audience. Correct any grammatical errors and improve the flow of the article. Ensure the output is in markdown format.", context={"draft": draft})
    #print(edit)
    blog_draft = edit
    # Get all LLM calls
    llm_calls = thread.get_llm_calls()
    print(f"LLM calls: {llm_calls}")

    # Get usage statistics
    usage = thread.get_usage()
    print(f"Usage: {usage}")

# write the draft to a markdown file named test.md
with open('test.md', 'w') as f:
    f.write(blog_draft)
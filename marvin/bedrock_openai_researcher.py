# Import necessary libraries
import boto3
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.providers.bedrock import BedrockProvider
import marvin
from marvin import Agent, Task

# Create a new AWS session using a specific profile
session = boto3.session.Session(profile_name='aws_sol')

# Create a Bedrock client using the session
bedrock_client = session.client('bedrock-runtime')

# Initialize a Bedrock model with a specific model name and provider
bedrock_model = BedrockConverseModel(
    model_name="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    provider=BedrockProvider(bedrock_client=bedrock_client)
)

# Define an OpenAI model identifier
openai_model = "openai:gpt-5-2025-08-07"

# Comment explaining the purpose of the code
# In this code, we'll use Bedrock for creation and research while OpenAI for editing

# Create an agent for research using the Bedrock model
researcher = Agent(
    model=bedrock_model,
    name="researcher",
    description="A researcher who can search the web for information",
    instructions="Perform a thorough research on the given topic and return the results in a structured format. If required, you can use multiple sources to gather information."
)

# Create an agent for writing using the Bedrock model
writer = Agent(
    model=bedrock_model,
    name="writer",
    description="A writer who can write a blog post",
    instructions="Write an article suitable for moderately technical audience. Use analogies when the technology is hard to explain. The output should be in markdown format. Don't use more than 3 analogies. Keep the blog article less than 1000 words."
)

# Create an agent for editing using the OpenAI model
editor = Agent(
    model=openai_model,
    name="editor",
    description="An editor who can edit a blog post",
    instructions="Edit the draft to make it suitable for the target audience. Correct any grammatical errors and improve the flow of the article. Ensure the output is in markdown format."
)

# Initialize a variable to store the blog draft
blog_draft = None

# Use a thread to manage the sequence of operations
with marvin.Thread() as thread:
    # Perform research using the researcher agent
    research = researcher.run("Analyze the current state of fusion research and the feasibility of using it for generating commercial electricity.")
    # Write a draft using the writer agent
    draft = writer.run("Write an article suitable for moderately technical audience. Use analogies when the technology is hard to explain. The output should be in markdown format.", context={"research": research})
    # Edit the draft using the editor agent
    edit = editor.run("Edit the draft to make it suitable for the target audience. Correct any grammatical errors and improve the flow of the article. Ensure the output is in markdown format.", context={"draft": draft})
    # Store the edited draft
    blog_draft = edit
    # Get all LLM calls made during the thread
    llm_calls = thread.get_llm_calls()
    print(f"LLM calls: {llm_calls}")

    # Get usage statistics for the thread
    usage = thread.get_usage()
    print(f"Usage: {usage}")

# Write the final blog draft to a markdown file named test.md
with open('test.md', 'w') as f:
    f.write(blog_draft)
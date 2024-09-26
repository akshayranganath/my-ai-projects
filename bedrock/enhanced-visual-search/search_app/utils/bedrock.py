import boto3
import json
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv
import os
from io import BytesIO

# load the environment variables
load_dotenv()

def create_session():
    # create am AWS session
    session = boto3.session.Session(profile_name='aws_sol')
    # create an instance of Bedrock Embedding
    bedrock = session.client( os.environ.get('BEDROCK_CLIENT') )
    return (session,bedrock)

def get_text_embedding(bedrock_session:str, text:str, model:str=os.environ.get('BEDROCK_EMBED_MODEL')):
    embedding = []      
    if text:
        # Request to get embedding
        response = bedrock_session.invoke_model(
            modelId=model,
            body=json.dumps({"inputText": text}),
            contentType="application/json",
        )
        
        # Parse the response to get the embedding
        response_body = json.loads(response['body'].read())
        embedding = response_body['embedding']
        
    return embedding
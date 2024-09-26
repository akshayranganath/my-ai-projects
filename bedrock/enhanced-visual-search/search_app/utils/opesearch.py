
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def create_opensearch_client(session):    
    # create the database index
    host = os.environ.get('OPENSEARCH_HOST')
    region = os.environ.get('AWS_REGION')
    service = os.environ.get('SERVICE_NAME')
    credentials = session.get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

    client = OpenSearch(
        hosts = [{'host': host, 'port': 443}],
        http_auth = awsauth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    return client

def vector_search(client, query_vector, top_k=5):
    index_name = os.environ.get('OPENSEARCH_INDEX')
    # Construct the search query
    search_query = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {  # Make sure this matches your field name
                    "vector": query_vector,
                    "k": top_k
                }
            }
        }
    }
    
    # Perform the search
    response = client.search(
        body=search_query,
        index=index_name
    )
    
    return response['hits']['hits']

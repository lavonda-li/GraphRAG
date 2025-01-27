import weaviate
from weaviate import Client as WeaviateClient
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env
# Access variables from .env
wcd_url = os.getenv("WCD_URL")  # Weaviate Cloud Deployment URL
wcd_api_key = os.getenv("WCD_API_KEY")  # Weaviate Cloud Deployment API key
openai_api_key = os.getenv("OPENAI_API_KEY")  # OpenAI API key

# Initialize Weaviate Client
def initialize_weaviate_client():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,
        auth_credentials=Auth.api_key(wcd_api_key),
        headers={'X-OpenAI-Api-key': openai_api_key}
    )
    return client

# Function to query Weaviate for Articles
def query_weaviate_articles(client, query_text, limit=10):
    # Perform vector search on Article collection
    response = client.collections.get("Article").query.near_text(
        query=query_text,
        limit=limit,
        return_metadata=MetadataQuery(distance=True)
    )

    # Parse response
    results = []
    for obj in response.objects:
        results.append({
            "uuid": obj.uuid,
            "properties": obj.properties,
            "distance": obj.metadata.distance,
        })
    return results

# Function to query Weaviate for MeSH Terms
def query_weaviate_terms(client, query_text, limit=10):
    # Perform vector search on MeshTerm collection
    response = client.collections.get("term").query.near_text(
        query=query_text,
        limit=limit,
        return_metadata=MetadataQuery(distance=True)
    )

    # Parse response
    results = []
    for obj in response.objects:
        results.append({
            "uuid": obj.uuid,
            "properties": obj.properties,
            "distance": obj.metadata.distance,
        })
    return results

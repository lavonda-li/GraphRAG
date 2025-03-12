import os
import re
from typing import Any

import pandas as pd
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

from extractor import GraphRAGExtractor
from query import GraphRAGQueryEngine
from store import GraphRAGStore

from IPython.core.display import Markdown


def parse_fn(response_str: str) -> Any:
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    return entities, relationships

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity")

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as ("relationship")

3. When finished, output.

-Real Data-
######################
text: {text}
######################
output:
"""


if __name__ == "__main__":
    # Step 1
    # Load sample dataset
    print("Step 1: Load sample dataset")
    news = pd.read_csv("https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv")[:50]

    # Step 2
    # Convert data into LlamaIndex Document objects
    print("Step 2: Convert data into LlamaIndex Document objects")
    documents = [
        Document(text=f"{row['title']}: {row['text']}")
        for _, row in news.iterrows()
    ]

    # Step 3
    print("Step 3: Split documents into nodes")
    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=20,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    # Step 4
    print("Step 4: Initialize LLM")
    OpenAI.api_key = os.getenv("OPENAI_API_KEY")
    print(OpenAI.api_key)
    llm = OpenAI(model="gpt-4o")

    entity_pattern = r'entity_name:\s*(.+?)\s*entity_type:\s*(.+?)\s*entity_description:\s*(.+?)\s*'
    relationship_pattern = r'source_entity:\s*(.+?)\s*target_entity:\s*(.+?)\s*relation:\s*(.+?)\s*relationship_description:\s*(.+?)\s*'

    kg_extractor = GraphRAGExtractor(
        llm=llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=2,
        parse_fn=parse_fn,
    )


    # Step 5
    print("Step 5: Create PropertyGraphIndex")
    index = PropertyGraphIndex(
        nodes=nodes,
        property_graph_store=GraphRAGStore(),
        kg_extractors=[kg_extractor],
        show_progress=True,
        embed_model="text-embedding-3-small"
    )

    # Step 6
    print("Step 6: Build communities")
    index.property_graph_store.build_communities() # type: ignore

    # Step 7
    print("Step 7: Query the graph")
    query_engine = GraphRAGQueryEngine(
        graph_store=index.property_graph_store, llm=llm
    )
    response = query_engine.query("What are news related to financial sector?")
    print(Markdown(f"{response.response}"))
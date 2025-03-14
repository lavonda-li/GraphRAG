import os
import pandas as pd
import json
import numpy as np
import voyageai
from openai import OpenAI

# Retrieve the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key == "OPENAI_API_KEY":
    raise ValueError("OPENAI_API_KEY not found or is still the placeholder in your environment variables.")

client = OpenAI(api_key=api_key)
from sklearn.metrics.pairwise import cosine_similarity

########################################################
# 1. LOADERS
########################################################

def load_graph_json(graphml_json_path):
    with open(graphml_json_path, 'r') as file:
        graph_json = json.load(file)
    nodes = graph_json["graphDataset"]["nodeRenderingData"]
    return {node_id: data["label"] for node_id, data in nodes.items()}

def load_embeddings(embedding_csv_path):
    df = pd.read_csv(embedding_csv_path)
    # Convert the stored 'embedding' from string => np.ndarray
    embeddings = df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=",")
    )
    df["embedding"] = list(embeddings)
    return df

def load_patient_summaries(patient_csv_path):
    df = pd.read_csv(patient_csv_path)
    return df["summary"]

########################################################
# 2. VOYAGE CLIENT
########################################################

vo = voyageai.Client()

########################################################
# 3. EMBEDDING FUNCTIONS
########################################################

def embed_document(text: str) -> np.ndarray:
    """Embed knowledge-base text (document)."""
    result = vo.embed([text], model="voyage-3", input_type="document")
    emb = np.array(result.embeddings[0], dtype=np.float32)
    return emb

def embed_query(query: str) -> np.ndarray:
    """Embed user query (input_type='query')."""
    result = vo.embed([query], model="voyage-3", input_type="query")
    emb = np.array(result.embeddings[0], dtype=np.float32)
    return emb

########################################################
# 4. K-NEAREST NEIGHBOR SEARCH
########################################################

def k_nearest_neighbors(query_emb: np.ndarray, doc_embs: np.ndarray, k: int = 5):
    """
    Returns the top-k document embeddings and their indices, based on dot-product similarity.
    (Voyage embeddings are normalized => dot product ~ cosine similarity.)

    doc_embs: shape (N, dim)
    query_emb: shape (dim, )

    Returns (retrieved_embds, retrieved_indices)
    """
    sims = np.dot(doc_embs, query_emb)
    top_k_indices = np.argsort(sims)[::-1][:k]
    retrieved_embds = doc_embs[top_k_indices]
    return retrieved_embds, top_k_indices

########################################################
# 5. SEARCH AND INTERSECTION (with Knowledge Graph)
########################################################

def query_embeddings_and_graph(
    patient_query: str,
    knowledge_embeddings: pd.Series,  # each item is a NumPy array
    knowledge_df: pd.DataFrame,
    node_labels: dict,
    top_n=5
):
    # 1) embed the query
    q_emb = embed_query(patient_query)

    # 2) build a stack of doc_embs from knowledge_embeddings (Series => shape (N, dim))
    doc_embs = np.stack(knowledge_embeddings.values)

    # 3) do k-nearest neighbor with dot-product
    _, retrieved_indices = k_nearest_neighbors(q_emb, doc_embs, k=top_n)

    # 4) retrieve the top-matching rows from the DF
    top_matches = knowledge_df.iloc[retrieved_indices]

    # 5) intersect with graph node names
    graph_node_names = set(node_labels.values())
    intersected = top_matches[top_matches["name"].isin(graph_node_names)]

    return intersected

########################################################
# 6. SUMMARY
########################################################

def generate_summary(patient_summary: str, intersected_matches: pd.DataFrame) -> str:
    if intersected_matches.empty:
        return (
            f"Patient Summary: {patient_summary}\n\n"
            "No intersecting knowledge base entries found.\n"
        )

    summary = (
        f"Patient Summary: {patient_summary}\n\n"
        "Relevant Knowledge Base Context:\n"
    )
    for _, row in intersected_matches.iterrows():
        summary += f"- {row['name']}: {row['description']}\n"
    return summary

########################################################
# 7. OPENAI API FOR PATIENT BRIEFINGS
########################################################

# Set your OpenAI API key here
  # Replace with your actual API key

def compile_patient_briefing(discharge_note: str, retrieval_summary: str) -> str:
    """
    Use OpenAI API to compile a patient briefing by combining the discharge note and the retrieval results.
    The briefing answers:
    1. Brief patient medical history/background
    2. Indication for echocardiography study/clinical question to answer
    3. Prior echocardiography study (yes/no; if yes, when)
    4. What should be focused on in the current study
    """
    prompt = f"""
You are a medical expert. Using the information provided below, compile a detailed patient briefing that includes answers to the following questions:
1. Brief patient medical history/background:
2. Indication for echocardiography study/clinical question to answer:
3. Prior echocardiography study (yes/no; if yes, when):
4. What should be focused on in the current study:

Discharge Note:
{discharge_note}

Additional Context:
{retrieval_summary}

Please structure your response clearly with headings for each question.
    """
    response = client.chat.completions.create(model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful and knowledgeable medical assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)
    briefing = response.choices[0].message.content.strip()
    return briefing

########################################################
# 8. MAIN WITH PATIENT BRIEFING
########################################################

def main_with_briefing(graphml_json_path, embedding_csv_path, patient_csv_path):
    node_labels = load_graph_json(graphml_json_path)
    knowledge_df = load_embeddings(embedding_csv_path)
    knowledge_embeddings = knowledge_df["embedding"]
    patient_summaries = load_patient_summaries(patient_csv_path)

    for patient_summary in patient_summaries:
        # Use each patient summary as the discharge note input
        intersected_matches = query_embeddings_and_graph(
            patient_query=patient_summary,
            knowledge_embeddings=knowledge_embeddings,
            knowledge_df=knowledge_df,
            node_labels=node_labels,
            top_n=5
        )
        retrieval_summary = generate_summary(patient_summary, intersected_matches)

        # Generate the final patient briefing using the OpenAI API
        final_briefing = compile_patient_briefing(patient_summary, retrieval_summary)
        print("============================================")
        print(final_briefing)

if __name__ == "__main__":
    graphml_json_path = "data/magi_knowledge_graph_20250312_221629.graphml.json"
    embedding_csv_path = "data/entities.csv"
    patient_csv_path = "data/RAG_test_input.csv"

    main_with_briefing(graphml_json_path, embedding_csv_path, patient_csv_path)

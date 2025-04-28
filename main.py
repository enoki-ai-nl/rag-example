from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_text_splitters import CharacterTextSplitter

from embeddings import get_embeddings

import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

loader = WikipediaLoader(query="Jaffa cake")

# Load the documents
docs = loader.load()

# Display the summary of the document
print("Summary: {} (Source: {})".format(docs[0].metadata['summary'], docs[0].metadata['source']))

# Split the text into chunks
text = ' '.join([page.page_content.replace('\\t', ' ') for page in docs])
text_splitter = CharacterTextSplitter(
   separator="\\n",
   chunk_size=1000,
   chunk_overlap=50,
   length_function=len,
   is_separator_regex=False,
)

texts = text_splitter.create_documents([text])
splits = [item.page_content for item in texts]

# Your API info
BENTO_EMBEDDING_MODEL_END_POINT = os.getenv("BENTO_EMBEDDING_MODEL_END_POINT")
BENTO_API_TOKEN = os.getenv("BENTO_API_TOKEN")

all_embeddings = []
# Pass the splits in a batch of 25
for i in range(0, len(splits), 25):
   batch = splits[i:i+25]
   # Pass the batch to the get_embeddings method
   embeddings_batch = get_embeddings(batch)
   # Append the embeddings to the all_embeddings list holdng the embeddings of the whole dataset
   all_embeddings.extend(embeddings_batch)

# Print the embeddings of the first 5 splits
print(all_embeddings[:5])

# Save the embeddings and splits to a DataFrame
import pandas as pd
df = pd.DataFrame({
   'page_content': splits,
   'embeddings': all_embeddings
})
print(df.head())

# Zilliz cluster credentials
user = os.getenv("user")
password = os.getenv("password")

# [example]
# uri = https://in03-ecebf22e0fa90a7.serverless.gcp-us-west1.cloud.zilliz.com
# token = db_admin:password (or ApiKey)

from pymilvus import MilvusClient

milvus_uri = "https://in03-ecebf22e0fa90a7.serverless.gcp-us-west1.cloud.zilliz.com"

milvus_client = MilvusClient(
    uri=milvus_uri,
    token="f02e87f2c44fd85bdd9ed56711b5dc3fedd6ae32ccfbbbddee048f050b8dfc7e6813d0106d049143fc3d8f860bd1110f6c6a55bc"
)
print(f"Connected to DB: {milvus_uri} successfully")

from pymilvus import MilvusClient, DataType
import numpy as np

# Collection name
collection_name = "RAG"

# Check and drop existing collection if needed
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)
    print(f"Dropped existing collection {collection_name}")

# Create schema
schema = milvus_client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("page_content", DataType.VARCHAR, max_length=65535)
schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=384)  # Adjust dimension to match your model

# Prepare index parameters
index_params = milvus_client.prepare_index_params()
index_params.add_index("embeddings", metric_type="COSINE")  # or "L2" depending on your needs

# Create collection with schema and index
milvus_client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params
)

def insert_data_to_milvus(client, collection_name, splits, all_embeddings):
    MAX_LENGTH = 65000  # Just below the limit to be safe
    rows = []
    
    for i in range(len(splits)):
        content = splits[i]
        # Check if content exceeds limit and truncate if needed
        if len(content) > MAX_LENGTH:
            print(f"Warning: Truncating content at index {i} from {len(content)} to {MAX_LENGTH} characters")
            content = content[:MAX_LENGTH]
        
        rows.append({
            "id": i,
            "page_content": content,
            "embeddings": all_embeddings[i]
        })
    
    # Continue with batch insertion as before
    batch_size = 100
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        client.insert(collection_name, batch)
        print(f"Batch {i//batch_size + 1}/{(len(rows) + batch_size - 1)//batch_size} inserted.")
    
    # Flush to ensure data is committed
    client.flush(collection_name)
    
    return len(rows)

# Search function for retrieving relevant documents
def get_relevant_docs(client, collection_name, user_query, top_k=8):
    # Get query embedding
    query_embedding = get_embeddings([user_query])[0]
    
    # Search parameters
    search_params = {
        "metric_type": "COSINE",  # Match your index metric type
        "params": {"nprobe": 10}  # Adjust based on your index type
    }
    
    # Perform the search
    results = client.search(
        collection_name=collection_name,
        data=[query_embedding],  # Note: needs to be in a list since it accepts multiple queries
        anns_field="embeddings",
        search_params=search_params,
        limit=top_k,
        output_fields=["page_content"]
    )
    
    # Combine results into a single document
    relevant_docs = " "
    for hit in results[0]:  # First query's results
        relevant_docs += hit["entity"]["page_content"]
    
    return relevant_docs

# STEP 5: Insert the data into Milvus
insert_data_to_milvus(milvus_client, collection_name, splits, all_embeddings)

# Example usage:
message = "What is a jaffa cake?"
relevant_docs = get_relevant_docs(milvus_client, collection_name, message, 1)

import bentoml

llm_client = bentoml.SyncHTTPClient(BENTO_EMBEDDING_MODEL_END_POINT, token=BENTO_API_TOKEN)

def dorag(question: str, context: str):
   # Define the prompt template
   prompt = (f"You are a helpful assistant. The user has a question. Answer the user question based only on the context: {context}. \\n"
             f"The user question is {question}")
   # Call the LLM endpoint with the prompt defined above
   results = llm_client.generate(
       max_tokens=1024,
       prompt=prompt,
   )
   res = ""
   for result in results:
       res += result
   return res

query = "What are cakes similar to jaffa cakes?"
print(dorag(question=query, context=relevant_docs))
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_text_splitters import CharacterTextSplitter
from embeddings import get_embeddings

import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load variables from .env file
load_dotenv()

# Get Milvus credentials from environment
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-ecebf22e0fa90a7.serverless.gcp-us-west1.cloud.zilliz.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

# Validate critical environment variables
if not MILVUS_TOKEN:
    logger.error("MILVUS_TOKEN is not set in the environment variables")
    sys.exit(1)

# Load documents from Wikipedia
logger.info("Loading documents from Wikipedia...")
loader = WikipediaLoader(query="Jaffa cake")
docs = loader.load()
logger.info(f"Loaded {len(docs)} documents")

# Display the summary of the document
if docs:
    logger.info(f"Summary: {docs[0].metadata.get('summary', 'No summary')} (Source: {docs[0].metadata.get('source', 'Unknown')})")

# Split the text into chunks
text = ' '.join([page.page_content.replace('\t', ' ') for page in docs])
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([text])
splits = [item.page_content for item in texts]
logger.info(f"Split text into {len(splits)} chunks")

# Generate embeddings using the local SentenceTransformer model
try:
    logger.info("Generating embeddings using SentenceTransformer...")
    # Use batching to avoid memory issues
    all_embeddings = []
    batch_size = 25
    
    # Generate embeddings in batches
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(splits) + batch_size - 1)//batch_size}")
        embeddings_batch = get_embeddings(batch)
        all_embeddings.extend(embeddings_batch)
    
    logger.info(f"Generated embeddings for {len(all_embeddings)} chunks")
except Exception as e:
    logger.error(f"Error generating embeddings: {e}")
    sys.exit(1)

# Create a DataFrame to store the splits and embeddings
import pandas as pd
df = pd.DataFrame({
    'page_content': splits,
    'embeddings': all_embeddings
})
logger.info(f"Created DataFrame with {len(df)} rows")

# Connect to Milvus
try:
    from pymilvus import MilvusClient
    
    logger.info(f"Connecting to Milvus at {MILVUS_URI}...")
    milvus_client = MilvusClient(
        uri=MILVUS_URI,
        token=MILVUS_TOKEN
    )
    logger.info(f"Connected to Milvus successfully")
    
    from pymilvus import DataType
    
    # Collection name
    collection_name = "RAG"
    
    # Check and drop existing collection if needed
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
        logger.info(f"Dropped existing collection {collection_name}")
    
    # Create schema
    schema = milvus_client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("page_content", DataType.VARCHAR, max_length=65535)
    
    # Get the embedding dimension from the first embedding
    embedding_dim = len(all_embeddings[0]) if all_embeddings else 384
    schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=embedding_dim)
    
    # Prepare index parameters
    index_params = milvus_client.prepare_index_params()
    index_params.add_index("embeddings", metric_type="COSINE")
    
    # Create collection with schema and index
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    logger.info(f"Created collection {collection_name}")
    
except Exception as e:
    logger.error(f"Error connecting to Milvus: {e}")
    sys.exit(1)

def insert_data_to_milvus(client, collection_name, splits, all_embeddings):
    MAX_LENGTH = 65000  # Just below the limit to be safe
    rows = []
    
    for i in range(len(splits)):
        content = splits[i]
        # Check if content exceeds limit and truncate if needed
        if len(content) > MAX_LENGTH:
            logger.warning(f"Truncating content at index {i} from {len(content)} to {MAX_LENGTH} characters")
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
        logger.info(f"Batch {i//batch_size + 1}/{(len(rows) + batch_size - 1)//batch_size} inserted.")
    
    # Flush to ensure data is committed
    client.flush(collection_name)
    
    return len(rows)

# Search function for retrieving relevant documents
def get_relevant_docs(client, collection_name, user_query, top_k=8):
    try:
        # Generate embedding for the query using the same embedding function
        logger.info(f"Generating embedding for query: {user_query}")
        query_embedding = get_embeddings([user_query])[0]
        
        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # Perform the search
        logger.info(f"Searching Milvus for top {top_k} relevant documents")
        results = client.search(
            collection_name=collection_name,
            data=[query_embedding],
            anns_field="embeddings",
            search_params=search_params,
            limit=top_k,
            output_fields=["page_content"]
        )
        
        # Combine results into a single document
        relevant_docs = ""
        if results and len(results) > 0 and len(results[0]) > 0:
            for hit in results[0]:  # First query's results
                relevant_docs += hit["entity"]["page_content"] + " "
            logger.info(f"Found {len(results[0])} relevant documents")
        else:
            logger.info("No search results returned")
        
        return relevant_docs
    except Exception as e:
        logger.error(f"Error searching for relevant documents: {e}")
        return ""

# Insert the data into Milvus
total_inserted = insert_data_to_milvus(milvus_client, collection_name, splits, all_embeddings)
logger.info(f"Inserted {total_inserted} documents into Milvus")

# Function to answer using the retrieved content
def answer_from_context(question, context):
    try:
        from transformers import pipeline
        
        # Use a QA pipeline as a simple way to answer questions
        logger.info("Initializing QA pipeline...")
        qa_pipeline = pipeline("question-answering")
        
        # Use the pipeline to answer the question
        if context:
            logger.info("Generating answer from context...")
            answer = qa_pipeline(question=question, context=context[:4000])  # Limit context to avoid token limit issues
            return answer["answer"]
        else:
            return "I couldn't find relevant information to answer that question."
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        # Install transformers if not present
        if "No module named 'transformers'" in str(e):
            logger.info("Installing transformers package...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
            try:
                from transformers import pipeline
                qa_pipeline = pipeline("question-answering")
                if context:
                    answer = qa_pipeline(question=question, context=context[:4000])
                    return answer["answer"]
            except Exception as new_e:
                logger.error(f"Error after installing transformers: {new_e}")
        
        # Return a portion of the context if we can't use the QA pipeline
        if context:
            return f"Here's what I found: {context[:500]}..."
        else:
            return "I couldn't find relevant information to answer that question."

# Example queries
example_queries = [
"How many calories are in a jaffa cake?",
    "What are cakes similar to jaffa cakes?",
    "What is a jaffa cake?",
    "Are jaffa cakes cookies or cakes?",

]

for query in example_queries:
    logger.info(f"\nProcessing query: {query}")
    
    # Get relevant documents
    relevant_docs = get_relevant_docs(milvus_client, collection_name, query, top_k=3)
    
    if relevant_docs:
        logger.info(f"Found relevant documents ({len(relevant_docs)} characters)")
        
        # Generate an answer
        answer = answer_from_context(query, relevant_docs)
        logger.info(f"Answer: {answer}")
    else:
        logger.info("No relevant documents found")
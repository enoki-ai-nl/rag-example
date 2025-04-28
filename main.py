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

# Check for required environment variables and provide defaults
BENTO_EMBEDDING_MODEL_END_POINT = os.getenv("BENTO_EMBEDDING_MODEL_END_POINT")
BENTO_API_TOKEN = os.getenv("BENTO_API_TOKEN")
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
text = ' '.join([page.page_content.replace('\\t', ' ') for page in docs])
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

# Get embeddings with fallback
logger.info("Generating embeddings...")
try:
    all_embeddings = []
    # Pass the splits in a batch of 25
    for i in range(0, len(splits), 25):
        batch = splits[i:i+25]
        # Try to use BentoML if configured, otherwise fall back to local SentenceTransformer
        embeddings_batch = get_embeddings(batch, BENTO_EMBEDDING_MODEL_END_POINT, BENTO_API_TOKEN)
        all_embeddings.extend(embeddings_batch)
    logger.info(f"Generated embeddings for {len(all_embeddings)} chunks")
except Exception as e:
    logger.error(f"Error generating embeddings: {e}")
    logger.info("Falling back to local SentenceTransformer model")
    all_embeddings = get_embeddings(splits)  # This will use the local model as fallback

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
        # Get query embedding - use the same method as for document embeddings
        query_embedding = get_embeddings([user_query], BENTO_EMBEDDING_MODEL_END_POINT, BENTO_API_TOKEN)[0]
        
        # Search parameters
        search_params = {
            "metric_type": "COSINE",  # Match your index metric type
            "params": {"nprobe": 10}  # Adjust based on your index type
        }
        
        # Perform the search
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
        for hit in results[0]:  # First query's results
            relevant_docs += hit["entity"]["page_content"] + " "
        
        return relevant_docs
    except Exception as e:
        logger.error(f"Error searching for relevant documents: {e}")
        return ""

# Insert the data into Milvus
total_inserted = insert_data_to_milvus(milvus_client, collection_name, splits, all_embeddings)
logger.info(f"Inserted {total_inserted} documents into Milvus")

# Function to generate response using LLM
def generate_response(question, context):
    try:
        import bentoml
        
        # Check if we have BentoML credentials
        if BENTO_EMBEDDING_MODEL_END_POINT and BENTO_API_TOKEN:
            logger.info("Using BentoML for response generation")
            llm_client = bentoml.SyncHTTPClient(BENTO_EMBEDDING_MODEL_END_POINT, token=BENTO_API_TOKEN)
            
            # Define the prompt template
            prompt = (f"You are a helpful assistant. The user has a question. Answer the user question based only on the context: {context}. \n"
                     f"The user question is {question}")
            
            # Call the LLM endpoint with the prompt
            results = llm_client.generate(
                max_tokens=1024,
                prompt=prompt,
            )
            
            res = ""
            for result in results:
                res += result
            return res
        else:
            logger.warning("BentoML credentials not available for response generation")
            return f"I found relevant information but couldn't generate a response. Here's what I found: {context[:500]}..."
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Sorry, I couldn't generate a response due to an error. Here's what I found: {context[:500]}..."

# Example query
example_queries = [
    "What are cakes similar to jaffa cakes?",
    "How many calories are in a jaffa cake?",
    "What are the main ingredients of a jaffa cake?",
]

for query in example_queries:
    logger.info(f"\nProcessing query: {query}")
    relevant_docs = get_relevant_docs(milvus_client, collection_name, query, top_k=3)
    
    if relevant_docs:
        logger.info(f"Found relevant documents ({len(relevant_docs)} characters)")
        
        # Try to generate a response if BentoML is available
        if BENTO_EMBEDDING_MODEL_END_POINT and BENTO_API_TOKEN:
            try:
                response = generate_response(query, relevant_docs)
                logger.info(f"Generated response: {response}")
            except Exception as e:
                logger.error(f"Failed to generate response: {e}")
                logger.info(f"Relevant context: {relevant_docs[:500]}...")
        else:
            logger.info("BentoML not configured, skipping response generation")
            logger.info(f"Relevant context: {relevant_docs[:500]}...")
    else:
        logger.info("No relevant documents found")
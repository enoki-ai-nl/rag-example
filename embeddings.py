# Import the libraries
import subprocess
import sys
import numpy as np

# Install the packages if the API key isn't provided
def install(package):
   subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# Define the embedding method
def get_embeddings(texts: list, BENTO_EMBEDDING_MODEL_END_POINT=None, BENTO_API_TOKEN=None) -> list:
   # If the BentoML KEY is provided, the method will use BENTOML model to get embeddings
   if BENTO_EMBEDDING_MODEL_END_POINT and BENTO_API_TOKEN:
       import bentoml
       embedding_client = bentoml.SyncHTTPClient(BENTO_EMBEDDING_MODEL_END_POINT, token=BENTO_API_TOKEN)
       return embedding_client.encode(sentences=texts).tolist()
  
   # Otherwise it'll use transformers library
   else:
       # Install transformers and torch if not already installed
       try:
           import transformers
       except ImportError:
           install("transformers")
       try:
           import torch
       except ImportError:
           install("torch")
      
       from transformers import AutoTokenizer, AutoModel
      
       # Initialize the tokenizer and model for embeddings
       tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
       model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
      
       inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
       with torch.no_grad():
           outputs = model(**inputs)
       embeddings = outputs.last_hidden_state.mean(dim=1)
       return embeddings.numpy().tolist()
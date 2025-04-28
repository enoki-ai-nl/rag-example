# Import the libraries
import subprocess
import sys
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Install missing packages if needed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Define the embedding method to only use SentenceTransformer
def get_embeddings(texts: list, BENTO_EMBEDDING_MODEL_END_POINT=None, BENTO_API_TOKEN=None) -> list:
    """
    Generate embeddings using the SentenceTransformer model.
    
    Args:
        texts: List of text strings to generate embeddings for
        
    Returns:
        List of embeddings as numpy arrays
    """
    logger.info(f"Generating embeddings for {len(texts)} texts using SentenceTransformer")
    
    # Install transformers and torch if not already installed
    try:
        import transformers
    except ImportError:
        logger.info("Installing transformers package")
        install("transformers")
        import transformers
    
    try:
        import torch
    except ImportError:
        logger.info("Installing torch package")
        install("torch")
        import torch
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Use SentenceTransformer directly for better efficiency
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    except ImportError:
        # Fall back to manual implementation if sentence_transformers not installed
        logger.info("SentenceTransformer not found, installing")
        install("sentence-transformers")
        
        try:
            # Try again with the package installed
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error using SentenceTransformer: {e}")
            
            # Last resort: use transformers directly
            logger.info("Falling back to using transformers directly")
            from transformers import AutoTokenizer, AutoModel
            
            logger.info("Initializing tokenizer and model")
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
            # Process in smaller batches to avoid OOM errors
            batch_size = 8  # Smaller batch size for safety
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                # Mean pooling to get sentence embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.extend(embeddings.numpy().tolist())
            
            return all_embeddings
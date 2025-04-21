import pinecone
from typing import Dict, List, Union, Optional, Any
import logging
import json
import os
import time
from tqdm import tqdm
import numpy as np
import redis
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeManager:
    def __init__(self, api_key: str, environment: str = "gcp-starter", use_cache: bool = True):
        """
        Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            use_cache: Whether to use Redis cache for frequently accessed vectors
        """
        if not api_key:
            raise ValueError("Pinecone API key is required")
            
        # Initialize Pinecone with the new API (v3.0.0)
        self.pc = pinecone.Pinecone(api_key=api_key)
        self.index_name = "physics-textbook"
        self.use_cache = use_cache
        self.redis_client = None
        
        if self.use_cache:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis cache: {str(e)}")
                self.redis_client = None
                self.use_cache = False
        
        # Create index if it doesn't exist
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # CLIP embedding dimension
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "gcp",
                            "region": "us-central1"
                        }
                    }
                )
                # Wait for index to be ready
                time.sleep(10)
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise

    def _cache_key(self, key: str) -> str:
        """Create a standardized cache key."""
        return f"pinecone:{self.index_name}:{key}"

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from Redis cache if available."""
        if not self.use_cache or not self.redis_client:
            return None
            
        try:
            data = self.redis_client.get(self._cache_key(key))
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Error getting from cache: {str(e)}")
        return None

    def _set_in_cache(self, key: str, data: Any, ttl: int = 3600):
        """Set data in Redis cache with TTL in seconds."""
        if not self.use_cache or not self.redis_client:
            return
            
        try:
            serialized = pickle.dumps(data)
            self.redis_client.setex(self._cache_key(key), ttl, serialized)
        except Exception as e:
            logger.warning(f"Error setting in cache: {str(e)}")

    def upsert_embeddings(self, chapter_data: Dict, batch_size: int = 100) -> None:
        """
        Upload embeddings to Pinecone.
        
        Args:
            chapter_data: Dictionary containing chapter embeddings
            batch_size: Size of batches for upserting
        """
        if not chapter_data or 'embeddings' not in chapter_data:
            raise ValueError("Invalid chapter data format")
            
        vectors = []
        
        # Process text embeddings
        for i, item in enumerate(chapter_data['embeddings']['text_embeddings']):
            if not item.get('embedding'):
                continue
            vector_id = f"{chapter_data['chapter_name']}_text_{i}"
            vectors.append({
                'id': vector_id,
                'values': item['embedding'],
                'metadata': {
                    'content': item['content'],
                    'type': item['type'],
                    'page': item['page'],
                    'chapter': chapter_data['chapter_name'],
                    'content_type': 'text'
                }
            })
        
        # Process image embeddings
        for i, item in enumerate(chapter_data['embeddings']['image_embeddings']):
            if not item.get('embedding'):
                continue
            vector_id = f"{chapter_data['chapter_name']}_image_{i}"
            vectors.append({
                'id': vector_id,
                'values': item['embedding'],
                'metadata': {
                    'image_path': item['path'],
                    'type': item['type'],
                    'page': item['page'],
                    'ocr_text': item['ocr_text'],
                    'chapter': chapter_data['chapter_name'],
                    'content_type': 'image'
                }
            })
        
        if not vectors:
            logger.warning("No valid vectors to upsert")
            return
            
        # Upsert in batches
        total_batches = (len(vectors) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading vectors", total=total_batches):
            batch = vectors[i:i + batch_size]
            
            # Retry logic for network failures
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.index.upsert(vectors=batch)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to upload batch after {max_retries} attempts: {str(e)}")
                        raise
                    else:
                        logger.warning(f"Error uploading batch (attempt {attempt+1}/{max_retries}): {str(e)}")
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            logger.debug(f"Uploaded batch of {len(batch)} vectors")
        
        logger.info(f"Successfully uploaded {len(vectors)} vectors for chapter {chapter_data['chapter_name']}")

    def search(self, 
              query_embedding: List[float], 
              filter: Optional[Dict] = None,
              top_k: int = 5,
              include_values: bool = False) -> List[Dict]:
        """
        Search for similar vectors in Pinecone.
        
        Args:
            query_embedding: Query vector
            filter: Optional metadata filters
            top_k: Number of results to return
            include_values: Whether to include vector values in results
            
        Returns:
            List of matching results with metadata
        """
        if not query_embedding:
            raise ValueError("Query embedding is required")
            
        # Create cache key from query
        cache_key = f"search:{hash(str(query_embedding))}-{hash(str(filter))}-{top_k}"
        cached_results = self._get_from_cache(cache_key)
        if cached_results:
            return cached_results
            
        try:
            # Normalize embedding if needed
            query_embedding_np = np.array(query_embedding)
            norm = np.linalg.norm(query_embedding_np)
            if norm > 0:
                normalized_embedding = query_embedding_np / norm
                query_embedding = normalized_embedding.tolist()
                
            results = self.index.query(
                vector=query_embedding,
                filter=filter,
                top_k=top_k,
                include_metadata=True,
                include_values=include_values
            )
            
            formatted_results = [{
                'score': match.score,
                'metadata': match.metadata,
                'id': match.id
            } for match in results.matches]
            
            # Cache the results
            self._set_in_cache(cache_key, formatted_results)
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            return []

    def bulk_search(self, query_embeddings: List[List[float]], filter: Optional[Dict] = None, top_k: int = 5) -> List[List[Dict]]:
        """
        Perform multiple searches in batch.
        
        Args:
            query_embeddings: List of query vectors
            filter: Optional metadata filters
            top_k: Number of results to return for each query
            
        Returns:
            List of search results for each query
        """
        if not query_embeddings:
            return []
            
        all_results = []
        for embedding in query_embeddings:
            try:
                results = self.search(embedding, filter, top_k)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Error in bulk search: {str(e)}")
                all_results.append([])
                
        return all_results

    def delete_chapter(self, chapter_name: str) -> None:
        """
        Delete all vectors for a specific chapter.
        
        Args:
            chapter_name: Name of the chapter to delete
        """
        if not chapter_name:
            raise ValueError("Chapter name is required")
            
        try:
            self.index.delete(
                filter={'chapter': chapter_name}
            )
            logger.info(f"Deleted vectors for chapter: {chapter_name}")
        except Exception as e:
            logger.error(f"Error deleting chapter vectors: {str(e)}")
            raise

    def get_stats(self) -> Dict:
        """
        Get index statistics.
        
        Returns:
            Dictionary containing index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'namespaces': stats.namespaces,
                'index_fullness': stats.total_vector_count / 1000000 if hasattr(stats, 'total_vector_count') else 0
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}
            
    def close(self):
        """Close any open connections."""
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Closed Redis connection")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {str(e)}")
                
    def __del__(self):
        """Destructor to ensure connections are closed."""
        self.close() 
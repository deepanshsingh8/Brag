import pinecone
from typing import Dict, List, Union, Optional
import logging
import json
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeManager:
    def __init__(self, api_key: str, environment: str = "gcp-starter"):
        """
        Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
        """
        if not api_key:
            raise ValueError("Pinecone API key is required")
            
        # Initialize Pinecone with the new API (v3.0.0)
        self.pc = pinecone.Pinecone(api_key=api_key)
        self.index_name = "physics-textbook"
        
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

    def upsert_embeddings(self, chapter_data: Dict) -> None:
        """
        Upload embeddings to Pinecone.
        
        Args:
            chapter_data: Dictionary containing chapter embeddings
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
            
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
                logger.info(f"Uploaded batch of {len(batch)} vectors")
            except Exception as e:
                logger.error(f"Error uploading batch to Pinecone: {str(e)}")
                raise

    def search(self, 
              query_embedding: List[float], 
              filter: Optional[Dict] = None,
              top_k: int = 5) -> List[Dict]:
        """
        Search for similar vectors in Pinecone.
        
        Args:
            query_embedding: Query vector
            filter: Optional metadata filters
            top_k: Number of results to return
            
        Returns:
            List of matching results with metadata
        """
        if not query_embedding:
            raise ValueError("Query embedding is required")
            
        try:
            results = self.index.query(
                vector=query_embedding,
                filter=filter,
                top_k=top_k,
                include_metadata=True
            )
            
            return [{
                'score': match.score,
                'metadata': match.metadata
            } for match in results.matches]
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            return []

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
                'dimension': stats.dimension
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {} 
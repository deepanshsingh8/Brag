from sentence_transformers import SentenceTransformer
from roboflow import Roboflow
import torch
from typing import Dict, List, Union
import os
import logging
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, roboflow_api_key: str):
        """
        Initialize the embedding generator with necessary models.
        
        Args:
            roboflow_api_key: API key for Roboflow
        """
        if not roboflow_api_key:
            raise ValueError("Roboflow API key is required")
            
        try:
            # Initialize text embedding model
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize Roboflow for CLIP embeddings
            self.rf = Roboflow(api_key=roboflow_api_key)
            
            logger.info("Embedding models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding models: {str(e)}")
            raise

    def generate_text_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for text content.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        if not text:
            logger.warning("Empty text provided for embedding")
            return []
            
        try:
            # Convert text to embedding
            embedding = self.text_model.encode(text)
            
            # Convert numpy array to list and ensure it's the correct type
            embedding_list = embedding.tolist()
            
            # Validate embedding
            if not isinstance(embedding_list, list) or not all(isinstance(x, float) for x in embedding_list):
                raise ValueError("Invalid embedding format")
                
            return embedding_list
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            return []

    def generate_image_embedding(self, image_path: str) -> List[float]:
        """
        Generate CLIP embeddings for images using Roboflow.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of embedding values
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return []
            
        try:
            # Load and validate image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get CLIP embeddings through Roboflow
            project = self.rf.workspace().project("physics-textbook")
            model = project.version(1).model
            
            # Get embeddings
            embedding = model.get_embeddings(image_path)
            
            # Validate embedding
            if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
                raise ValueError("Invalid embedding format from Roboflow")
                
            return embedding
        except Exception as e:
            logger.error(f"Error generating image embedding: {str(e)}")
            return []
        finally:
            if 'image' in locals():
                image.close()

    def process_chapter_content(self, chapter_data: Dict) -> Dict[str, List[Dict]]:
        """
        Process all content from a chapter and generate embeddings.
        
        Args:
            chapter_data: Dictionary containing chapter content
            
        Returns:
            Dictionary with text and image embeddings
        """
        if not chapter_data:
            raise ValueError("Chapter data is required")
            
        embeddings = {
            'text_embeddings': [],
            'image_embeddings': []
        }
        
        try:
            # Process text blocks
            for block in chapter_data.get('text_blocks', []):
                if not block.get('content'):
                    continue
                    
                embedding = self.generate_text_embedding(block['content'])
                if embedding:
                    embeddings['text_embeddings'].append({
                        'content': block['content'],
                        'type': block.get('type', 'text'),
                        'page': block.get('page', 0),
                        'embedding': embedding
                    })
            
            # Process image blocks
            for block in chapter_data.get('image_blocks', []):
                if not block.get('path'):
                    continue
                    
                embedding = self.generate_image_embedding(block['path'])
                if embedding:
                    embeddings['image_embeddings'].append({
                        'path': block['path'],
                        'type': block.get('type', 'general_image'),
                        'page': block.get('page', 0),
                        'ocr_text': block.get('ocr_text', ''),
                        'embedding': embedding
                    })
            
            return embeddings
        except Exception as e:
            logger.error(f"Error processing chapter content: {str(e)}")
            raise

    def batch_process_chapters(self, chapters_data: List[Dict]) -> List[Dict]:
        """
        Process multiple chapters in batch.
        
        Args:
            chapters_data: List of chapter content dictionaries
            
        Returns:
            List of processed chapters with embeddings
        """
        if not chapters_data:
            raise ValueError("Chapters data is required")
            
        processed_chapters = []
        
        for chapter_data in chapters_data:
            try:
                embeddings = self.process_chapter_content(chapter_data)
                processed_chapters.append({
                    'chapter_name': chapter_data.get('chapter_name', 'unknown'),
                    'embeddings': embeddings
                })
            except Exception as e:
                logger.error(f"Error processing chapter: {str(e)}")
                continue
        
        return processed_chapters 
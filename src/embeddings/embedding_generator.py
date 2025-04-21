from sentence_transformers import SentenceTransformer
from roboflow import Roboflow
import torch
from typing import Dict, List, Union, Optional, Callable
import os
import logging
from PIL import Image
import numpy as np
import concurrent.futures
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, roboflow_api_key: str, 
                model_name: str = "all-MiniLM-L6-v2",
                use_parallel: bool = True,
                max_workers: int = 4):
        """
        Initialize the embedding generator with necessary models.
        
        Args:
            roboflow_api_key: API key for Roboflow
            model_name: Name of the sentence transformer model to use
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of workers for parallel processing
        """
        if not roboflow_api_key:
            raise ValueError("Roboflow API key is required")
            
        try:
            # Save configuration
            self.use_parallel = use_parallel
            self.max_workers = max_workers
            
            # Initialize text embedding model
            self.text_model = SentenceTransformer(model_name)
            self.model_name = model_name
            
            # Initialize Roboflow for CLIP embeddings
            self.rf = Roboflow(api_key=roboflow_api_key)
            
            # Check for CUDA availability
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                self.text_model = self.text_model.to(self.device)
                logger.info(f"Using CUDA for embeddings generation on {self.device}")
            else:
                logger.info("Using CPU for embeddings generation")
            
            logger.info(f"Embedding models initialized successfully with model: {model_name}")
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
            if not isinstance(embedding_list, list) or not all(isinstance(x, (float, np.float32, np.float64)) for x in embedding_list):
                raise ValueError("Invalid embedding format")
                
            return embedding_list
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            return []

    def generate_batch_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a batch.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding lists
        """
        if not texts:
            return []
            
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and isinstance(text, str)]
            if not valid_texts:
                return []
                
            # Generate embeddings in batch
            embeddings = self.text_model.encode(valid_texts)
            
            # Convert numpy arrays to lists
            embedding_lists = embeddings.tolist()
            
            return embedding_lists
        except Exception as e:
            logger.error(f"Error generating batch text embeddings: {str(e)}")
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
            if not isinstance(embedding, list) or not all(isinstance(x, (float, np.float32, np.float64)) for x in embedding):
                raise ValueError("Invalid embedding format from Roboflow")
                
            return embedding
        except Exception as e:
            logger.error(f"Error generating image embedding: {str(e)}")
            return []
        finally:
            if 'image' in locals():
                image.close()

    def process_text_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Process multiple text blocks with parallel option."""
        if not blocks:
            return []
            
        result = []
        
        if self.use_parallel and len(blocks) > 1:
            # Extract text content for batch processing
            contents = [block.get('content', '') for block in blocks if block.get('content')]
            
            # Generate embeddings in batch
            start_time = time.time()
            embeddings = self.generate_batch_text_embeddings(contents)
            logger.info(f"Generated {len(embeddings)} text embeddings in {time.time() - start_time:.2f} seconds")
            
            # Match embeddings back to original blocks
            embedding_idx = 0
            for block in blocks:
                if not block.get('content'):
                    continue
                    
                if embedding_idx < len(embeddings):
                    embedding = embeddings[embedding_idx]
                    embedding_idx += 1
                    
                    result.append({
                        'content': block['content'],
                        'type': block.get('type', 'text'),
                        'page': block.get('page', 0),
                        'embedding': embedding
                    })
        else:
            # Process sequentially
            for block in tqdm(blocks, desc="Processing text blocks"):
                if not block.get('content'):
                    continue
                    
                embedding = self.generate_text_embedding(block['content'])
                if embedding:
                    result.append({
                        'content': block['content'],
                        'type': block.get('type', 'text'),
                        'page': block.get('page', 0),
                        'embedding': embedding
                    })
                    
        return result

    def process_image_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Process multiple image blocks with parallel option."""
        if not blocks:
            return []
            
        result = []
        
        if self.use_parallel and len(blocks) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create a mapping of future to original block
                future_to_block = {}
                
                for block in blocks:
                    if not block.get('path'):
                        continue
                        
                    future = executor.submit(self.generate_image_embedding, block['path'])
                    future_to_block[future] = block
                
                # Process results as they complete
                for future in tqdm(concurrent.futures.as_completed(future_to_block), 
                                  total=len(future_to_block), 
                                  desc="Processing images"):
                    block = future_to_block[future]
                    try:
                        embedding = future.result()
                        if embedding:
                            result.append({
                                'path': block['path'],
                                'type': block.get('type', 'general_image'),
                                'page': block.get('page', 0),
                                'ocr_text': block.get('ocr_text', ''),
                                'embedding': embedding
                            })
                    except Exception as e:
                        logger.error(f"Error processing image {block.get('path')}: {str(e)}")
        else:
            # Process sequentially
            for block in tqdm(blocks, desc="Processing image blocks"):
                if not block.get('path'):
                    continue
                    
                embedding = self.generate_image_embedding(block['path'])
                if embedding:
                    result.append({
                        'path': block['path'],
                        'type': block.get('type', 'general_image'),
                        'page': block.get('page', 0),
                        'ocr_text': block.get('ocr_text', ''),
                        'embedding': embedding
                    })
                    
        return result

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
            
        start_time = time.time()
        logger.info(f"Starting processing of chapter: {chapter_data.get('chapter_name', 'unknown')}")
        
        embeddings = {
            'text_embeddings': [],
            'image_embeddings': []
        }
        
        try:
            # Process text blocks
            text_blocks = chapter_data.get('text_blocks', [])
            if text_blocks:
                logger.info(f"Processing {len(text_blocks)} text blocks")
                embeddings['text_embeddings'] = self.process_text_blocks(text_blocks)
            
            # Process image blocks
            image_blocks = chapter_data.get('image_blocks', [])
            if image_blocks:
                logger.info(f"Processing {len(image_blocks)} image blocks")
                embeddings['image_embeddings'] = self.process_image_blocks(image_blocks)
            
            processing_time = time.time() - start_time
            logger.info(f"Completed processing in {processing_time:.2f} seconds: " +
                       f"{len(embeddings['text_embeddings'])} text embeddings, " +
                       f"{len(embeddings['image_embeddings'])} image embeddings")
            
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
        total_chapters = len(chapters_data)
        
        for idx, chapter_data in enumerate(chapters_data, 1):
            try:
                logger.info(f"Processing chapter {idx}/{total_chapters}: {chapter_data.get('chapter_name', 'unknown')}")
                embeddings = self.process_chapter_content(chapter_data)
                processed_chapters.append({
                    'chapter_name': chapter_data.get('chapter_name', 'unknown'),
                    'embeddings': embeddings
                })
            except Exception as e:
                logger.error(f"Error processing chapter: {str(e)}")
                continue
        
        return processed_chapters
        
    def get_model_info(self) -> Dict:
        """Get information about the embedding models."""
        return {
            "text_model": self.model_name,
            "model_dimension": len(self.generate_text_embedding("test sample")),
            "device": str(self.device),
            "parallel_processing": self.use_parallel,
            "max_workers": self.max_workers
        } 
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import torch
from typing import Dict, List, Union
import logging
from ..vector_store.pinecone_manager import PineconeManager
from ..embeddings.embedding_generator import EmbeddingGenerator
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionGenerator:
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: PineconeManager):
        """
        Initialize the question generator.
        
        Args:
            embedding_generator: Instance of EmbeddingGenerator
            vector_store: Instance of PineconeManager
        """
        if not embedding_generator or not vector_store:
            raise ValueError("Embedding generator and vector store are required")
            
        try:
            # Load question generation model
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
            self.model = AutoModelForSeq2SeqGeneration.from_pretrained("google/flan-t5-large")
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.embedding_generator = embedding_generator
            self.vector_store = vector_store
            
            logger.info("Question generator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing question generator: {str(e)}")
            raise

    def generate_similar_questions(self, input_question: str, num_questions: int = 5) -> List[Dict]:
        """
        Generate similar questions based on input question and relevant content.
        
        Args:
            input_question: The seed question
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions with context
        """
        if not input_question or not isinstance(input_question, str):
            raise ValueError("Valid input question is required")
            
        if num_questions < 1 or num_questions > 10:
            raise ValueError("Number of questions must be between 1 and 10")
            
        try:
            # Get question embedding
            question_embedding = self.embedding_generator.generate_text_embedding(input_question)
            if not question_embedding:
                raise ValueError("Failed to generate question embedding")
            
            # Search for relevant content
            search_results = self.vector_store.search(
                query_embedding=question_embedding,
                top_k=3
            )
            
            if not search_results:
                logger.warning("No relevant content found for question generation")
                return []
            
            generated_questions = []
            
            for result in search_results:
                context = result['metadata'].get('content', '')
                if not context:
                    continue
                    
                # Generate question prompt
                prompt = f"""
                Context: {context}
                
                Original Question: {input_question}
                
                Task: Generate a similar but different physics question based on the context and original question.
                Make sure the new question:
                1. Tests similar concepts
                2. Has different specific values or scenarios
                3. Is clear and well-formed
                4. Is appropriate for Class 12 physics level
                5. Includes relevant units and measurements
                6. Can be answered using the given context
                
                Generated Question:
                """
                
                # Generate new questions
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                
                try:
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs["input_ids"],
                            max_length=128,
                            num_return_sequences=min(2, num_questions),
                            temperature=0.7,
                            top_p=0.95,
                            do_sample=True
                        )
                    
                    for output in outputs:
                        question = self.tokenizer.decode(output, skip_special_tokens=True)
                        # Clean and validate the generated question
                        question = self._clean_question(question)
                        if question:
                            generated_questions.append({
                                'question': question,
                                'context': context,
                                'similarity_score': result['score'],
                                'source_page': result['metadata'].get('page', 0),
                                'chapter': result['metadata'].get('chapter', 'unknown')
                            })
                except Exception as e:
                    logger.error(f"Error generating question: {str(e)}")
                    continue
            
            # Sort by similarity score and return top num_questions
            generated_questions.sort(key=lambda x: x['similarity_score'], reverse=True)
            return generated_questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Error in question generation: {str(e)}")
            return []

    def _clean_question(self, question: str) -> str:
        """Clean and validate the generated question."""
        if not question:
            return ""
            
        # Remove any prompt artifacts
        question = re.sub(r'^Generated Question:|^Question:|^\d+\.\s*', '', question)
        question = question.strip()
        
        # Basic validation
        if len(question) < 10 or not question.endswith('?'):
            return ""
            
        return question

    def analyze_question_difficulty(self, question: str) -> str:
        """
        Analyze the difficulty level of a question.
        
        Args:
            question: The question to analyze
            
        Returns:
            Difficulty level (Easy, Medium, Hard)
        """
        if not question:
            return "Medium"
            
        prompt = f"""
        Question: {question}
        
        Task: Analyze the difficulty level of this physics question for Class 12 students.
        Consider:
        1. Conceptual complexity
        2. Mathematical complexity
        3. Number of steps required
        4. Integration of multiple concepts
        5. Required prior knowledge
        6. Problem-solving approach needed
        
        Difficulty Level (choose one: Easy, Medium, Hard):
        """
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=16,
                    num_return_sequences=1,
                    temperature=0.3
                )
            
            difficulty = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Validate difficulty level
            difficulty = difficulty.lower()
            if 'easy' in difficulty:
                return "Easy"
            elif 'hard' in difficulty:
                return "Hard"
            return "Medium"
            
        except Exception as e:
            logger.error(f"Error analyzing question difficulty: {str(e)}")
            return "Medium" 
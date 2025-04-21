from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import torch
from typing import Dict, List, Union, Optional
import logging
from ..vector_store.pinecone_manager import PineconeManager
from ..embeddings.embedding_generator import EmbeddingGenerator
import re
import os
from dotenv import load_dotenv
import json
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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
            # Initialize components
            self.embedding_generator = embedding_generator
            self.vector_store = vector_store
            
            # Load OpenAI API key
            openai_api_key = os.getenv('OPENAI_API_KEY')
            
            # Setup LLM (fallback to FLAN-T5 if OpenAI key not available)
            if openai_api_key:
                self.use_openai = True
                self.llm = ChatOpenAI(
                    model_name="gpt-4o",
                    temperature=0.7,
                    openai_api_key=openai_api_key
                )
                
                # Define templates for different question types
                self.question_template = PromptTemplate(
                    input_variables=["context", "original_question", "question_type", "difficulty"],
                    template="""
                    You are an expert physics teacher creating questions for Class 12 physics students.
                    
                    Context: {context}
                    
                    Original Question: {original_question}
                    
                    Task: Generate a {difficulty} {question_type} physics question based on the context.
                    The question should:
                    1. Test similar physics concepts as the original
                    2. Use different specific values, scenarios, or formulations
                    3. Be appropriate for Class 12 physics level
                    4. Include relevant units and scientific notation where appropriate
                    5. Be challenging but solvable with the given context
                    
                    Respond with ONLY the generated question, nothing else.
                    """
                )
                
                # Chain for generating questions
                self.question_chain = LLMChain(
                    llm=self.llm,
                    prompt=self.question_template,
                    verbose=False
                )
                
                logger.info("Using OpenAI for question generation")
            else:
                self.use_openai = False
                # Fallback to local model
                self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
                self.model = AutoModelForSeq2SeqGeneration.from_pretrained("google/flan-t5-large")
                self.model.eval()
                logger.info("Using FLAN-T5 for question generation (OpenAI API key not found)")
            
            logger.info("Question generator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing question generator: {str(e)}")
            raise

    def generate_similar_questions(self, 
                                  input_question: str, 
                                  num_questions: int = 5, 
                                  question_types: Optional[List[str]] = None,
                                  difficulty: str = "Medium") -> List[Dict]:
        """
        Generate similar questions based on input question and relevant content.
        
        Args:
            input_question: The seed question
            num_questions: Number of questions to generate
            question_types: Types of questions to generate (conceptual, numerical, etc.)
            difficulty: Difficulty level of questions
            
        Returns:
            List of generated questions with context
        """
        if not input_question or not isinstance(input_question, str):
            raise ValueError("Valid input question is required")
            
        if num_questions < 1 or num_questions > 10:
            raise ValueError("Number of questions must be between 1 and 10")
            
        # Default question types if not provided
        if not question_types:
            question_types = ["conceptual", "numerical", "application", "analytical"]
        
        try:
            # Get question embedding
            question_embedding = self.embedding_generator.generate_text_embedding(input_question)
            if not question_embedding:
                raise ValueError("Failed to generate question embedding")
            
            # Search for relevant content
            search_results = self.vector_store.search(
                query_embedding=question_embedding,
                top_k=max(3, num_questions)  # Get more content for variety
            )
            
            if not search_results:
                logger.warning("No relevant content found for question generation")
                return []
            
            generated_questions = []
            
            # Distribute questions among content and question types
            for i in range(num_questions):
                # Select context in round-robin fashion
                result_idx = i % len(search_results)
                result = search_results[result_idx]
                
                # Select question type in round-robin fashion
                question_type = question_types[i % len(question_types)]
                
                # Get content
                context = result['metadata'].get('content', '')
                if not context and 'ocr_text' in result['metadata']:
                    context = result['metadata'].get('ocr_text', '')
                
                if not context:
                    continue
                
                if self.use_openai:
                    try:
                        # Generate question with OpenAI
                        response = self.question_chain.run(
                            context=context,
                            original_question=input_question,
                            question_type=question_type,
                            difficulty=difficulty
                        )
                        
                        question = self._clean_question(response)
                        if question:
                            generated_questions.append({
                                'question': question,
                                'context': context,
                                'similarity_score': result['score'],
                                'source_page': result['metadata'].get('page', 0),
                                'chapter': result['metadata'].get('chapter', 'unknown'),
                                'type': question_type,
                                'difficulty': difficulty
                            })
                    except Exception as e:
                        logger.error(f"Error generating question with OpenAI: {str(e)}")
                        # Fall back to FLAN-T5 if OpenAI fails
                        self._generate_with_flan_t5(
                            context, input_question, question_type, 
                            difficulty, result, generated_questions
                        )
                else:
                    # Generate with FLAN-T5
                    self._generate_with_flan_t5(
                        context, input_question, question_type, 
                        difficulty, result, generated_questions
                    )
            
            # Sort by similarity score
            generated_questions.sort(key=lambda x: x['similarity_score'], reverse=True)
            return generated_questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Error in question generation: {str(e)}")
            return []

    def _generate_with_flan_t5(self, context, input_question, question_type, difficulty, result, generated_questions):
        """Helper method to generate questions with FLAN-T5."""
        try:
            # Generate question prompt for FLAN-T5
            prompt = f"""
            Context: {context}
            
            Original Question: {input_question}
            
            Task: Generate a {difficulty} {question_type} physics question based on the context and original question.
            Make sure the new question:
            1. Tests similar concepts
            2. Has different specific values or scenarios
            3. Is clear and well-formed
            4. Is appropriate for Class 12 physics level
            5. Includes relevant units and measurements
            
            Generated Question:
            """
            
            # Generate new questions
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=128,
                    num_return_sequences=1,
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
                        'chapter': result['metadata'].get('chapter', 'unknown'),
                        'type': question_type,
                        'difficulty': difficulty
                    })
        except Exception as e:
            logger.error(f"Error generating with FLAN-T5: {str(e)}")

    def _clean_question(self, question: str) -> str:
        """Clean and validate the generated question."""
        if not question:
            return ""
            
        # Remove any prompt artifacts
        question = re.sub(r'^Generated Question:|^Question:|^\d+\.\s*', '', question)
        question = question.strip()
        
        # Basic validation
        if len(question) < 10:
            return ""
        
        # Add question mark if missing
        if not question.endswith('?'):
            question = question + '?'
            
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
        
        if self.use_openai:
            try:
                # Use OpenAI to analyze difficulty
                template = """
                Question: {question}
                
                Analyze the difficulty level of this physics question for Class 12 students.
                Consider:
                1. Conceptual complexity
                2. Mathematical complexity
                3. Number of steps required
                4. Integration of multiple concepts
                
                Return ONLY one word: "Easy", "Medium", or "Hard"
                """
                
                prompt = PromptTemplate(
                    input_variables=["question"],
                    template=template
                )
                
                chain = LLMChain(llm=self.llm, prompt=prompt)
                difficulty = chain.run(question=question).strip()
                
                # Normalize response
                difficulty = difficulty.lower()
                if 'easy' in difficulty:
                    return "Easy"
                elif 'hard' in difficulty:
                    return "Hard"
                return "Medium"
            except Exception as e:
                logger.error(f"Error analyzing difficulty with OpenAI: {str(e)}")
                # Fall back to FLAN-T5
                return self._analyze_with_flan_t5(question)
        else:
            # Use FLAN-T5
            return self._analyze_with_flan_t5(question)

    def _analyze_with_flan_t5(self, question: str) -> str:
        """Analyze question difficulty with FLAN-T5."""
        try:
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
            logger.error(f"Error analyzing difficulty with FLAN-T5: {str(e)}")
            return "Medium"
            
    def generate_quiz(self, topic: str, num_questions: int = 5, include_answers: bool = False) -> Dict:
        """
        Generate a complete quiz on a specific topic.
        
        Args:
            topic: The physics topic to generate questions about
            num_questions: Number of questions to include
            include_answers: Whether to include answers
            
        Returns:
            Dictionary containing quiz questions and metadata
        """
        if not topic or not isinstance(topic, str):
            raise ValueError("Valid topic is required")
            
        try:
            # Generate embedding for topic
            topic_embedding = self.embedding_generator.generate_text_embedding(topic)
            
            # Search for relevant content
            search_results = self.vector_store.search(
                query_embedding=topic_embedding,
                top_k=max(5, num_questions)
            )
            
            if not search_results:
                logger.warning(f"No relevant content found for topic: {topic}")
                return {'success': False, 'message': f"No content found for topic: {topic}"}
            
            # Generate a mix of question types and difficulties
            question_types = ["conceptual", "numerical", "application", "analytical"]
            difficulties = ["Easy", "Medium", "Hard"]
            
            questions = []
            # Ensure a mix of difficulties
            for i in range(num_questions):
                question_type = question_types[i % len(question_types)]
                difficulty = difficulties[i % len(difficulties)]
                
                # Create a seed question based on the topic
                seed_question = f"What is {topic} in physics and how does it work?"
                
                # Generate similar questions
                similar_questions = self.generate_similar_questions(
                    seed_question, 
                    num_questions=1,
                    question_types=[question_type],
                    difficulty=difficulty
                )
                
                if similar_questions:
                    questions.append(similar_questions[0])
            
            # Create quiz metadata
            quiz = {
                'title': f"Physics Quiz: {topic}",
                'topic': topic,
                'created_at': time.time(),
                'num_questions': len(questions),
                'questions': questions
            }
            
            return quiz
        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return {'success': False, 'message': f"Error generating quiz: {str(e)}"} 
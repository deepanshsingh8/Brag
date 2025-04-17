import gradio as gr
import os
from dotenv import load_dotenv
from ..embeddings.embedding_generator import EmbeddingGenerator
from ..vector_store.pinecone_manager import PineconeManager
from ..question_gen.generator import QuestionGenerator
from ..pdf_processor.extractor import PDFExtractor
import logging
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PhysicsQuestionApp:
    def __init__(self):
        """Initialize the application components."""
        try:
            # Initialize components with API keys
            self.embedding_generator = EmbeddingGenerator(
                roboflow_api_key=os.getenv('ROBOFLOW_API_KEY')
            )
            self.vector_store = PineconeManager(
                api_key=os.getenv('PINECONE_API_KEY')
            )
            self.question_generator = QuestionGenerator(
                embedding_generator=self.embedding_generator,
                vector_store=self.vector_store
            )
            
            # Create output directory
            self.output_dir = "data/processed"
            os.makedirs(self.output_dir, exist_ok=True)
            
            logger.info("Application initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing application: {str(e)}")
            raise

    def process_pdf(self, pdf_file) -> str:
        """Process a PDF file and store its content."""
        if not pdf_file:
            return "No PDF file provided"
            
        try:
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp()
            temp_pdf_path = os.path.join(temp_dir, pdf_file.name)
            
            try:
                # Save uploaded PDF temporarily
                with open(temp_pdf_path, 'wb') as f:
                    f.write(pdf_file.read())
                
                # Process PDF
                extractor = PDFExtractor(temp_pdf_path, self.output_dir)
                chapter_data = extractor.process_chapter()
                
                # Generate embeddings
                processed_data = self.embedding_generator.process_chapter_content(chapter_data)
                
                # Store in Pinecone
                self.vector_store.upsert_embeddings({
                    'chapter_name': chapter_data['chapter_name'],
                    'embeddings': processed_data
                })
                
                return f"Successfully processed {pdf_file.name} and stored in the database."
            finally:
                # Clean up temporary files
                shutil.rmtree(temp_dir)
                
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return f"Error processing PDF: {str(e)}"

    def generate_questions(self, input_question: str, num_questions: int) -> str:
        """Generate similar questions based on input."""
        if not input_question or not input_question.strip():
            return "Please enter a valid question."
            
        try:
            questions = self.question_generator.generate_similar_questions(
                input_question, 
                num_questions=num_questions
            )
            
            if not questions:
                return "No similar questions could be generated. Please try a different input question."
            
            # Format output
            output = []
            for i, q in enumerate(questions, 1):
                difficulty = self.question_generator.analyze_question_difficulty(q['question'])
                output.append(f"\nQuestion {i} (Difficulty: {difficulty}):")
                output.append(f"{q['question']}")
                output.append(f"\nSource: Chapter {q['chapter']}, Page {q['source_page']}")
                output.append("-" * 50)
            
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return f"Error generating questions: {str(e)}"

    def create_interface(self):
        """Create and launch the Gradio interface."""
        with gr.Blocks(title="Physics Question Generator", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # Physics Question Generator
            Upload NCERT Physics textbook PDFs and generate similar questions based on your input.
            """)
            
            with gr.Tab("Process PDF"):
                with gr.Row():
                    pdf_input = gr.File(
                        label="Upload Physics PDF",
                        file_types=[".pdf"],
                        type="binary"
                    )
                    process_button = gr.Button("Process PDF", variant="primary")
                
                process_output = gr.Textbox(
                    label="Processing Status",
                    lines=2,
                    interactive=False
                )
                
                process_button.click(
                    fn=self.process_pdf,
                    inputs=[pdf_input],
                    outputs=process_output
                )
            
            with gr.Tab("Generate Questions"):
                with gr.Row():
                    with gr.Column():
                        question_input = gr.Textbox(
                            label="Input Question",
                            placeholder="Enter a physics question...",
                            lines=3
                        )
                        num_questions = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Questions"
                        )
                        generate_button = gr.Button("Generate Similar Questions", variant="primary")
                    
                    question_output = gr.Textbox(
                        label="Generated Questions",
                        lines=15,
                        interactive=False
                    )
                
                generate_button.click(
                    fn=self.generate_questions,
                    inputs=[question_input, num_questions],
                    outputs=question_output
                )
        
        return interface

def main():
    try:
        app = PhysicsQuestionApp()
        interface = app.create_interface()
        interface.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860
        )
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
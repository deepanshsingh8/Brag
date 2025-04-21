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
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import io

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
            
            # Store processed PDFs for tracking
            self.processed_pdfs = []
            self._load_processed_pdfs()
            
            logger.info("Application initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing application: {str(e)}")
            raise

    def _load_processed_pdfs(self):
        """Load list of previously processed PDFs."""
        try:
            pdf_log_path = os.path.join(self.output_dir, "processed_pdfs.json")
            if os.path.exists(pdf_log_path):
                with open(pdf_log_path, 'r') as f:
                    self.processed_pdfs = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading processed PDFs: {str(e)}")
            self.processed_pdfs = []

    def _save_processed_pdfs(self):
        """Save list of processed PDFs."""
        try:
            pdf_log_path = os.path.join(self.output_dir, "processed_pdfs.json")
            with open(pdf_log_path, 'w') as f:
                json.dump(self.processed_pdfs, f)
        except Exception as e:
            logger.warning(f"Error saving processed PDFs: {str(e)}")

    def process_pdf(self, pdf_file, use_cache=True) -> str:
        """Process a PDF file and store its content."""
        if not pdf_file:
            return "No PDF file provided"
            
        try:
            start_time = time.time()
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp()
            temp_pdf_path = os.path.join(temp_dir, pdf_file.name)
            
            try:
                # Save uploaded PDF temporarily
                with open(temp_pdf_path, 'wb') as f:
                    f.write(pdf_file.read())
                
                # Process PDF
                extractor = PDFExtractor(temp_pdf_path, self.output_dir, use_cache=use_cache)
                chapter_data = extractor.process_chapter()
                
                # Generate embeddings
                processed_data = self.embedding_generator.process_chapter_content(chapter_data)
                
                # Store in Pinecone
                self.vector_store.upsert_embeddings({
                    'chapter_name': chapter_data['chapter_name'],
                    'embeddings': processed_data
                })
                
                # Record processed PDF
                pdf_info = {
                    "name": pdf_file.name,
                    "chapter_name": chapter_data['chapter_name'],
                    "timestamp": time.time(),
                    "text_blocks": len(chapter_data['text_blocks']),
                    "image_blocks": len(chapter_data['image_blocks'])
                }
                self.processed_pdfs.append(pdf_info)
                self._save_processed_pdfs()
                
                processing_time = time.time() - start_time
                
                return f"✅ Successfully processed {pdf_file.name} in {processing_time:.2f} seconds.\n" \
                       f"Extracted {len(chapter_data['text_blocks'])} text blocks and {len(chapter_data['image_blocks'])} images."
            finally:
                # Clean up temporary files
                shutil.rmtree(temp_dir)
                
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return f"❌ Error processing PDF: {str(e)}"

    def generate_questions(self, input_question: str, num_questions: int, question_type: str, difficulty: str) -> str:
        """Generate similar questions based on input."""
        if not input_question or not input_question.strip():
            return "Please enter a valid question."
            
        try:
            question_types = [question_type] if question_type != "all" else None
            
            questions = self.question_generator.generate_similar_questions(
                input_question, 
                num_questions=num_questions,
                question_types=question_types,
                difficulty=difficulty
            )
            
            if not questions:
                return "No similar questions could be generated. Please try a different input question."
            
            # Format output
            output = []
            for i, q in enumerate(questions, 1):
                output.append(f"\nQuestion {i} (Type: {q.get('type', 'general')}, Difficulty: {q.get('difficulty', 'Medium')}):")
                output.append(f"{q['question']}")
                output.append(f"\nSource: Chapter {q['chapter']}, Page {q['source_page']}")
                output.append("-" * 50)
            
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return f"Error generating questions: {str(e)}"

    def generate_quiz(self, topic: str, num_questions: int) -> str:
        """Generate a complete quiz on a physics topic."""
        if not topic or not topic.strip():
            return "Please enter a valid physics topic."
            
        try:
            quiz = self.question_generator.generate_quiz(
                topic=topic,
                num_questions=num_questions
            )
            
            if not quiz.get('questions'):
                return f"Could not generate quiz for topic '{topic}'. Please try a different topic."
            
            # Format output
            output = [f"# Physics Quiz: {topic}", ""]
            
            for i, q in enumerate(quiz['questions'], 1):
                output.append(f"\n## Question {i} (Type: {q.get('type', 'general')}, Difficulty: {q.get('difficulty', 'Medium')})")
                output.append(f"{q['question']}")
                output.append(f"\nSource: Chapter {q['chapter']}, Page {q['source_page']}")
                output.append("-" * 50)
            
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return f"Error generating quiz: {str(e)}"

    def search_content(self, search_query: str, content_type: str, num_results: int) -> tuple:
        """Search for content in the vector database."""
        if not search_query or not search_query.strip():
            return "Please enter a valid search query.", None
            
        try:
            # Generate embedding for search query
            query_embedding = self.embedding_generator.generate_text_embedding(search_query)
            
            # Add filter for content type if specified
            filter_dict = None
            if content_type != "all":
                filter_dict = {"content_type": content_type}
                
            # Search vector database
            results = self.vector_store.search(
                query_embedding=query_embedding,
                filter=filter_dict,
                top_k=num_results
            )
            
            if not results:
                return "No matching content found. Please try a different query.", None
            
            # Format output text
            output = [f"# Search Results for: '{search_query}'", ""]
            
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                score = result['score']
                
                output.append(f"\n## Result {i} (Similarity: {score:.4f})")
                
                if metadata.get('content_type') == 'text':
                    output.append(f"Type: {metadata.get('type', 'text')}")
                    output.append(f"Chapter: {metadata.get('chapter', 'unknown')}, Page: {metadata.get('page', 0)}")
                    output.append(f"\nContent: {metadata.get('content', 'No content available')}")
                else:
                    output.append(f"Type: Image ({metadata.get('type', 'general_image')})")
                    output.append(f"Chapter: {metadata.get('chapter', 'unknown')}, Page: {metadata.get('page', 0)}")
                    output.append(f"OCR Text: {metadata.get('ocr_text', 'No OCR text available')}")
                
                output.append("-" * 50)
            
            # Create visualization
            fig = self._create_search_visualization(results)
            
            return "\n".join(output), fig
        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
            return f"Error searching content: {str(e)}", None

    def _create_search_visualization(self, results):
        """Create visualization of search results."""
        try:
            # Extract scores and result types
            scores = [result['score'] for result in results]
            types = [result['metadata'].get('type', 'unknown') for result in results]
            content_types = [result['metadata'].get('content_type', 'unknown') for result in results]
            
            # Create type mapping for colors
            unique_types = list(set(types))
            type_colors = {t: plt.cm.tab10(i % 10) for i, t in enumerate(unique_types)}
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create bar chart
            bars = ax.bar(range(len(scores)), scores, color=[type_colors[t] for t in types])
            
            # Add markers for content type
            for i, (score, content_type) in enumerate(zip(scores, content_types)):
                marker = 'o' if content_type == 'text' else 's'
                ax.plot(i, score + 0.02, marker=marker, markersize=10, 
                       color='black', markerfacecolor='none')
            
            # Add labels and title
            ax.set_xlabel('Result Number')
            ax.set_ylabel('Similarity Score')
            ax.set_title('Search Results by Similarity')
            ax.set_xticks(range(len(scores)))
            ax.set_xticklabels([f"{i+1}" for i in range(len(scores))])
            
            # Add legend for types
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color, markersize=10, label=t)
                              for t, color in type_colors.items()]
            # Add legend for content types
            legend_elements.extend([
                plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='none', 
                          markersize=10, label='Text'),
                plt.Line2D([0], [0], marker='s', color='black', markerfacecolor='none', 
                          markersize=10, label='Image')
            ])
            
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Convert figure to image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            
            return img
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None

    def get_database_stats(self):
        """Get statistics about the vector database."""
        try:
            stats = self.vector_store.get_stats()
            pdf_count = len(self.processed_pdfs)
            
            # Format output
            output = [
                f"# Database Statistics",
                f"\nTotal vectors: {stats.get('total_vectors', 0):,}",
                f"Total PDFs processed: {pdf_count}",
                f"Vector dimension: {stats.get('dimension', 384)}",
                f"Index fullness: {stats.get('index_fullness', 0)*100:.2f}%"
            ]
            
            if pdf_count > 0:
                output.append("\n## Processed PDFs:")
                for i, pdf in enumerate(self.processed_pdfs[-10:], 1):  # Show last 10
                    output.append(f"{i}. {pdf['name']} - {pdf['text_blocks']} text blocks, {pdf['image_blocks']} images")
            
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return f"Error getting database stats: {str(e)}"

    def create_interface(self):
        """Create and launch the Gradio interface."""
        with gr.Blocks(title="Physics Question Generator", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # Physics Question Generator
            Upload NCERT Physics textbook PDFs, search for content, and generate questions and quizzes.
            """)
            
            with gr.Tab("Process PDF"):
                with gr.Row():
                    pdf_input = gr.File(
                        label="Upload Physics PDF",
                        file_types=[".pdf"],
                        type="binary"
                    )
                
                with gr.Row():
                    use_cache = gr.Checkbox(label="Use Cache", value=True)
                    process_button = gr.Button("Process PDF", variant="primary")
                
                process_output = gr.Textbox(
                    label="Processing Status",
                    lines=5,
                    interactive=False
                )
                
                process_button.click(
                    fn=self.process_pdf,
                    inputs=[pdf_input, use_cache],
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
                        with gr.Row():
                            num_questions = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Number of Questions"
                            )
                            question_type = gr.Dropdown(
                                choices=["all", "conceptual", "numerical", "application", "analytical"],
                                value="all",
                                label="Question Type"
                            )
                            difficulty = gr.Dropdown(
                                choices=["Easy", "Medium", "Hard"],
                                value="Medium",
                                label="Difficulty Level"
                            )
                        generate_button = gr.Button("Generate Similar Questions", variant="primary")
                    
                    question_output = gr.Textbox(
                        label="Generated Questions",
                        lines=15,
                        interactive=False
                    )
                
                generate_button.click(
                    fn=self.generate_questions,
                    inputs=[question_input, num_questions, question_type, difficulty],
                    outputs=question_output
                )
            
            with gr.Tab("Generate Quiz"):
                with gr.Row():
                    with gr.Column():
                        topic_input = gr.Textbox(
                            label="Physics Topic",
                            placeholder="Enter a physics topic (e.g., Electromagnetism, Optics, etc.)",
                            lines=1
                        )
                        quiz_num_questions = gr.Slider(
                            minimum=3,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Questions"
                        )
                        quiz_button = gr.Button("Generate Quiz", variant="primary")
                    
                    quiz_output = gr.Textbox(
                        label="Generated Quiz",
                        lines=20,
                        interactive=False
                    )
                
                quiz_button.click(
                    fn=self.generate_quiz,
                    inputs=[topic_input, quiz_num_questions],
                    outputs=quiz_output
                )
            
            with gr.Tab("Search Content"):
                with gr.Row():
                    with gr.Column():
                        search_input = gr.Textbox(
                            label="Search Query",
                            placeholder="Enter search terms...",
                            lines=1
                        )
                        with gr.Row():
                            content_type = gr.Dropdown(
                                choices=["all", "text", "image"],
                                value="all",
                                label="Content Type"
                            )
                            num_results = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Number of Results"
                            )
                        search_button = gr.Button("Search", variant="primary")
                
                with gr.Row():
                    search_output = gr.Textbox(
                        label="Search Results",
                        lines=15,
                        interactive=False
                    )
                    visualization = gr.Image(
                        label="Results Visualization",
                        type="pil"
                    )
                
                search_button.click(
                    fn=self.search_content,
                    inputs=[search_input, content_type, num_results],
                    outputs=[search_output, visualization]
                )
            
            with gr.Tab("Database Stats"):
                stats_button = gr.Button("Refresh Statistics", variant="primary")
                stats_output = gr.Textbox(
                    label="Database Statistics",
                    lines=15,
                    interactive=False
                )
                
                # Load stats when tab is opened
                stats_button.click(
                    fn=self.get_database_stats,
                    inputs=[],
                    outputs=stats_output
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
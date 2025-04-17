import os
from dotenv import load_dotenv
from pdf_processor.extractor import PDFExtractor
from embeddings.embedding_generator import EmbeddingGenerator
from vector_store.pinecone_manager import PineconeManager
import logging
import sys
from datetime import datetime
import traceback

# Configure logging with colors and formatting
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )

    def format(self, record):
        if record.levelno == logging.INFO:
            record.msg = f"{self.blue}{record.msg}{self.reset}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{self.yellow}{record.msg}{self.reset}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{self.red}{record.msg}{self.reset}"
        return super().format(record)

# Set up logging
logger = logging.getLogger("Physics-PDF-Processor")
logger.setLevel(logging.INFO)

# Console handler with color formatting
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter())
logger.addHandler(console_handler)

# File handler for keeping logs
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler(f"logs/processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def process_all_pdfs():
    """Process all physics PDFs and store their embeddings."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Validate environment variables
        if not os.getenv('PINECONE_API_KEY'):
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        if not os.getenv('ROBOFLOW_API_KEY'):
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set")
        
        logger.info("=" * 50)
        logger.info("Starting PDF Processing Pipeline")
        logger.info("=" * 50)
        
        # Initialize components
        logger.info("Initializing components...")
        try:
            embedding_generator = EmbeddingGenerator(
                roboflow_api_key=os.getenv('ROBOFLOW_API_KEY')
            )
            vector_store = PineconeManager(
                api_key=os.getenv('PINECONE_API_KEY')
            )
            logger.info("Components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
        
        # Create output directory
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each PDF
        pdf_dir = "src/ncert_chapters"
        if not os.path.exists(pdf_dir):
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
            
        pdf_files = [f for f in os.listdir(pdf_dir) if f.startswith('leph') and f.endswith('.pdf')]
        total_pdfs = len(pdf_files)
        
        if total_pdfs == 0:
            raise FileNotFoundError("No PDF files found in the directory")
        
        logger.info(f"Found {total_pdfs} PDFs to process")
        logger.info("-" * 50)
        
        successful_processing = 0
        failed_processing = 0
        
        for idx, pdf_file in enumerate(sorted(pdf_files), 1):
            try:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                logger.info(f"Processing PDF {idx}/{total_pdfs}: {pdf_file}")
                logger.info("Stage 1/3: Extracting content...")
                
                # Extract content
                extractor = PDFExtractor(pdf_path, output_dir)
                chapter_data = extractor.process_chapter()
                
                logger.info("Stage 2/3: Generating embeddings...")
                # Generate embeddings
                processed_data = embedding_generator.process_chapter_content(chapter_data)
                
                logger.info("Stage 3/3: Storing in Pinecone...")
                # Store in Pinecone
                vector_store.upsert_embeddings({
                    'chapter_name': chapter_data['chapter_name'],
                    'embeddings': processed_data
                })
                
                logger.info(f"✓ Successfully processed {pdf_file}")
                logger.info("-" * 50)
                successful_processing += 1
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                failed_processing += 1
                continue
        
        # Get final stats
        stats = vector_store.get_stats()
        logger.info("=" * 50)
        logger.info("Processing Pipeline Complete!")
        logger.info(f"Successfully processed: {successful_processing}/{total_pdfs} PDFs")
        logger.info(f"Failed to process: {failed_processing}/{total_pdfs} PDFs")
        logger.info(f"Total vectors in database: {stats.get('total_vectors', 0)}")
        logger.info("=" * 50)
        
        if failed_processing > 0:
            raise RuntimeError(f"Failed to process {failed_processing} PDFs")

    except Exception as e:
        logger.error(f"Fatal error in processing pipeline: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    try:
        process_all_pdfs()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1) 
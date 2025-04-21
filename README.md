# Physics Question Generator - Agentic RAG System

An intelligent system that processes physics textbooks (like NCERT) to generate relevant questions for teachers. The system uses advanced RAG (Retrieval Augmented Generation) techniques with parallel processing to handle both text and images from physics textbooks.

## Features

- **Advanced PDF Processing**: Extract text and images with caching for faster reprocessing
- **High-quality OCR**: Enhanced image processing for better text extraction
- **Parallel Embedding Generation**: Batch processing of text and concurrent image embedding generation
- **Vector Storage with Caching**: Redis-based caching for frequently accessed vectors in Pinecone
- **Multiple LLM Support**: OpenAI GPT models with FLAN-T5 fallback for question generation
- **Rich Question Generation**: Create different types of questions (conceptual, numerical, analytical)
- **Complete Quiz Generation**: Generate full quizzes on specific physics topics
- **Semantic Search**: Find relevant content across text and images with visualization
- **Modern UI**: User-friendly Gradio interface with multiple tabs and features

## Setup

1. Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/physics-question-generator.git
cd physics-question-generator
pip install -r requirements.txt
```

2. Set up external tools:
- **Tesseract OCR** for text extraction from images
  - MacOS: `brew install tesseract`
  - Ubuntu: `apt-get install tesseract-ocr`
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Redis** (optional for caching)
  - MacOS: `brew install redis`
  - Ubuntu: `apt-get install redis-server`
  - Windows: Download from [Redis website](https://redis.io/download)

3. Environment Setup:
Create a `.env` file with the following variables:
```
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
ROBOFLOW_API_KEY=your_roboflow_api_key
```

4. Create required directories:
```bash
mkdir -p data/processed/text
mkdir -p data/processed/images
mkdir -p data/processed/cache
```

## Project Structure

```
.
├── src/
│   ├── pdf_processor/     # Enhanced PDF text and image extraction with caching
│   ├── embeddings/        # Parallel text and image embedding generation
│   ├── vector_store/      # Pinecone vector database with Redis caching
│   ├── question_gen/      # Question generation with OpenAI/FLAN-T5
│   ├── ui/                # Improved Gradio interface with visualization
│   ├── process_all_pdfs.py # Batch processing script
│   └── run_ui.py          # Entry point to run the application
├── data/
│   └── processed/         # Processed data with caching
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation
```

## Usage

1. Run the Gradio UI:
```bash
python src/run_ui.py
```

2. Batch process PDFs:
```bash
python src/process_all_pdfs.py
```

## Features Detail

### PDF Processing
- Automatically extracts text and images from physics textbooks
- Uses enhanced OCR with preprocessing for better image text extraction
- Implements intelligent content type detection (formulas, questions, diagrams)
- Caches processed results to avoid redundant processing

### Embedding Generation
- Uses sentence-transformers for text content with batch processing
- Leverages CLIP embeddings via Roboflow for image content
- Parallel processing for large documents with progress tracking

### Vector Storage
- Pinecone vector database for efficient similarity search
- Redis caching layer for frequently accessed vectors
- Automatic retry logic and connection management

### Question Generation
- Uses OpenAI GPT models with fallback to FLAN-T5
- Generates varied question types (conceptual, numerical, application-based)
- Quiz generation for specific physics topics
- Difficulty level analysis and tagging

### User Interface
- Multiple tabs for different functions (processing, question generation, search)
- Real-time visualization of search results
- Database statistics and processing history
- Comprehensive error handling and user feedback

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
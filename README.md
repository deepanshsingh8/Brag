# Physics Question Generator - Agentic RAG System

An intelligent system that processes NCERT Physics textbooks to generate relevant questions for teachers. The system uses advanced RAG (Retrieval Augmented Generation) techniques to process both text and images from physics textbooks.

## Features

- PDF text and image extraction
- Image processing using CLIP embeddings via Roboflow
- Text embeddings using state-of-the-art language models
- Vector storage using Pinecone
- Question generation based on textbook content
- User-friendly Gradio interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Required external tools:
- Tesseract OCR for text extraction from images
- Poppler for PDF processing

3. Environment variables needed:
```
PINECONE_API_KEY=your_key
OPENAI_API_KEY=your_key
ROBOFLOW_API_KEY=your_key
```

## Project Structure

```
.
├── src/
│   ├── pdf_processor/      # PDF text and image extraction
│   ├── embeddings/         # Text and image embedding generation
│   ├── vector_store/       # Pinecone database management
│   ├── question_gen/       # Question generation logic
│   └── ui/                 # Gradio interface
├── data/
│   └── processed/          # Processed text and images
├── config/                 # Configuration files
└── notebooks/             # Development notebooks
```

## Usage

1. Process PDFs:
```bash
python src/process_pdfs.py
```

2. Generate embeddings:
```bash
python src/generate_embeddings.py
```

3. Run the Gradio interface:
```bash
python src/run_ui.py
``` 
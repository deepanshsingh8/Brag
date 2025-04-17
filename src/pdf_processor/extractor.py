import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging
from typing import Dict, List, Tuple
import re
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self, pdf_path: str, output_dir: str):
        """
        Initialize the PDF extractor.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted content
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.doc = None
        self.temp_dir = None
        
        try:
            # Create output directories if they don't exist
            self.text_dir = os.path.join(output_dir, "text")
            self.images_dir = os.path.join(output_dir, "images")
            os.makedirs(self.text_dir, exist_ok=True)
            os.makedirs(self.images_dir, exist_ok=True)
            
            # Create temporary directory for processing
            self.temp_dir = tempfile.mkdtemp()
            
            # Open PDF document
            self.doc = fitz.open(pdf_path)
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Error initializing PDF extractor: {str(e)}")

    def cleanup(self):
        """Clean up resources."""
        if self.doc:
            self.doc.close()
            self.doc = None
            
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary directory: {str(e)}")
            self.temp_dir = None

    def extract_text_and_images(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract text and images from the PDF.
        
        Returns:
            Tuple containing lists of text and image metadata
        """
        if not self.doc:
            raise RuntimeError("PDF document not initialized")
            
        text_blocks = []
        image_blocks = []
        
        try:
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    # Split text into sections (paragraphs, formulas, etc.)
                    sections = self._split_text_into_sections(text)
                    for section in sections:
                        text_blocks.append({
                            'page': page_num + 1,
                            'content': section,
                            'type': self._identify_content_type(section)
                        })
                
                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = self.doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Save image to temporary directory first
                        temp_image_path = os.path.join(
                            self.temp_dir, 
                            f"page_{page_num + 1}_img_{img_index + 1}.png"
                        )
                        with open(temp_image_path, 'wb') as img_file:
                            img_file.write(image_bytes)
                        
                        # Extract text from image using OCR
                        image = Image.open(temp_image_path)
                        ocr_text = pytesseract.image_to_string(image)
                        image.close()  # Close the image file
                        
                        # Move image to final location
                        final_image_path = os.path.join(
                            self.images_dir, 
                            f"page_{page_num + 1}_img_{img_index + 1}.png"
                        )
                        shutil.move(temp_image_path, final_image_path)
                        
                        image_blocks.append({
                            'page': page_num + 1,
                            'path': final_image_path,
                            'ocr_text': ocr_text.strip(),
                            'type': self._identify_image_type(ocr_text)
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing image on page {page_num + 1}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error extracting content from PDF: {str(e)}")
            raise
            
        return text_blocks, image_blocks

    def _split_text_into_sections(self, text: str) -> List[str]:
        """Split text into meaningful sections."""
        if not text:
            return []
            
        # Split by double newlines to separate paragraphs
        sections = [s.strip() for s in text.split('\n\n') if s.strip()]
        return sections

    def _identify_content_type(self, text: str) -> str:
        """Identify the type of text content."""
        if not text:
            return 'text'
            
        # Check for mathematical formulas
        if re.search(r'[=+\-×÷∫∑π√]', text):
            return 'formula'
        # Check for questions (starts with number or contains question words)
        elif re.match(r'^\d+[\.\)]', text) or re.search(r'\b(what|why|how|when|where|which)\b', text.lower()):
            return 'question'
        # Default to regular text
        return 'text'

    def _identify_image_type(self, ocr_text: str) -> str:
        """Identify the type of image based on OCR text."""
        if not ocr_text:
            return 'general_image'
            
        if re.search(r'[=+\-×÷∫∑π√]', ocr_text):
            return 'formula_image'
        elif re.search(r'fig|figure|diagram', ocr_text.lower()):
            return 'diagram'
        return 'general_image'

    def process_chapter(self) -> Dict:
        """
        Process the entire chapter and return structured content.
        
        Returns:
            Dict containing processed chapter content
        """
        try:
            text_blocks, image_blocks = self.extract_text_and_images()
            
            # Save text content
            chapter_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
            text_output_path = os.path.join(self.text_dir, f"{chapter_name}.txt")
            
            with open(text_output_path, 'w', encoding='utf-8') as f:
                for block in text_blocks:
                    f.write(f"--- {block['type'].upper()} (Page {block['page']}) ---\n")
                    f.write(block['content'])
                    f.write('\n\n')
            
            return {
                'chapter_name': chapter_name,
                'text_blocks': text_blocks,
                'image_blocks': image_blocks,
                'text_path': text_output_path
            }
        except Exception as e:
            logger.error(f"Error processing chapter: {str(e)}")
            raise
        finally:
            self.cleanup()

    def __del__(self):
        """Clean up resources."""
        self.cleanup() 
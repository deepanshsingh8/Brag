import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging
from typing import Dict, List, Tuple, Optional, Set
import re
import tempfile
import shutil
import hashlib
import json
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self, pdf_path: str, output_dir: str, use_cache: bool = True):
        """
        Initialize the PDF extractor.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted content
            use_cache: Whether to use caching for extracted content
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.use_cache = use_cache
        self.doc = None
        self.temp_dir = None
        
        # Generate a content hash for caching
        self.content_hash = self._generate_content_hash()
        self.cache_dir = os.path.join(output_dir, "cache")
        
        try:
            # Create output directories if they don't exist
            self.text_dir = os.path.join(output_dir, "text")
            self.images_dir = os.path.join(output_dir, "images")
            os.makedirs(self.text_dir, exist_ok=True)
            os.makedirs(self.images_dir, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Create temporary directory for processing
            self.temp_dir = tempfile.mkdtemp()
            
            # Open PDF document
            self.doc = fitz.open(pdf_path)
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Error initializing PDF extractor: {str(e)}")

    def _generate_content_hash(self) -> str:
        """Generate a hash of the PDF content for caching."""
        try:
            with open(self.pdf_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception:
            # If hashing fails, use the file name
            return os.path.basename(self.pdf_path)

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

    def _load_from_cache(self) -> Optional[Dict]:
        """Load processed content from cache if available."""
        if not self.use_cache:
            return None
            
        cache_file = os.path.join(self.cache_dir, f"{self.content_hash}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded from cache: {cache_file}")
                return data
            except Exception as e:
                logger.warning(f"Error loading cache: {str(e)}")
        return None

    def _save_to_cache(self, data: Dict):
        """Save processed content to cache."""
        if not self.use_cache:
            return
            
        cache_file = os.path.join(self.cache_dir, f"{self.content_hash}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")

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
            total_pages = len(self.doc)
            for page_num in tqdm(range(total_pages), desc="Extracting pages"):
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
                        
                        # Enhance image for better OCR
                        enhanced_image = self._enhance_image_for_ocr(temp_image_path)
                        
                        # Extract text from image using OCR with advanced settings
                        ocr_text = pytesseract.image_to_string(
                            enhanced_image,
                            config='--psm 6 --oem 3'  # Page segmentation mode 6, LSTM OCR engine 3
                        )
                        enhanced_image.close()  # Close the image file
                        
                        # Move image to final location
                        final_image_path = os.path.join(
                            self.images_dir, 
                            f"{self.content_hash}_page_{page_num + 1}_img_{img_index + 1}.png"
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

    def _enhance_image_for_ocr(self, image_path: str) -> Image.Image:
        """Enhance image for better OCR results."""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Increase image size for better OCR
            width, height = image.size
            if width < 1000 or height < 1000:
                scale_factor = 2
                image = image.resize((width * scale_factor, height * scale_factor), Image.LANCZOS)
                
            return image
        except Exception as e:
            logger.warning(f"Error enhancing image: {str(e)}")
            return Image.open(image_path)

    def _split_text_into_sections(self, text: str) -> List[str]:
        """Split text into meaningful sections."""
        if not text:
            return []
            
        # Split by double newlines to separate paragraphs
        sections = [s.strip() for s in text.split('\n\n') if s.strip()]
        
        # Further split long sections (more than 500 characters)
        result = []
        for section in sections:
            if len(section) > 500:
                # Split long sections by sentences
                subsections = re.split(r'(?<=[.!?])\s+', section)
                
                # Group sentences into chunks of reasonable size
                current_chunk = ""
                for subsection in subsections:
                    if len(current_chunk) + len(subsection) <= 500:
                        current_chunk += " " + subsection if current_chunk else subsection
                    else:
                        if current_chunk:
                            result.append(current_chunk.strip())
                        current_chunk = subsection
                
                if current_chunk:
                    result.append(current_chunk.strip())
            else:
                result.append(section)
                
        return result

    def _identify_content_type(self, text: str) -> str:
        """Identify the type of text content."""
        if not text:
            return 'text'
            
        # Check for mathematical formulas
        if re.search(r'[=+\-×÷∫∑π√]', text) or re.search(r'([a-zA-Z]+\^[0-9]+)', text):
            return 'formula'
        # Check for questions (starts with number or contains question words)
        elif re.match(r'^\d+[\.\)]', text) or re.search(r'\b(what|why|how|when|where|which)\b', text.lower()):
            return 'question'
        # Check for definition
        elif re.search(r'\b(defined as|definition|is called|refers to)\b', text.lower()):
            return 'definition'
        # Default to regular text
        return 'text'

    def _identify_image_type(self, ocr_text: str) -> str:
        """Identify the type of image based on OCR text."""
        if not ocr_text:
            return 'general_image'
            
        ocr_text = ocr_text.lower()
        
        if re.search(r'[=+\-×÷∫∑π√]', ocr_text):
            return 'formula_image'
        elif re.search(r'fig|figure|diagram', ocr_text):
            return 'diagram'
        elif re.search(r'table|tabular', ocr_text):
            return 'table'
        elif re.search(r'graph|plot|curve', ocr_text):
            return 'graph'
        elif re.search(r'circuit|schematic', ocr_text):
            return 'circuit'
        return 'general_image'

    def process_chapter(self) -> Dict:
        """
        Process the entire chapter and return structured content.
        
        Returns:
            Dict containing processed chapter content
        """
        try:
            # Check cache first
            cached_data = self._load_from_cache()
            if cached_data:
                return cached_data
                
            text_blocks, image_blocks = self.extract_text_and_images()
            
            # Save text content
            chapter_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
            text_output_path = os.path.join(self.text_dir, f"{chapter_name}.txt")
            
            with open(text_output_path, 'w', encoding='utf-8') as f:
                for block in text_blocks:
                    f.write(f"--- {block['type'].upper()} (Page {block['page']}) ---\n")
                    f.write(block['content'])
                    f.write('\n\n')
            
            result = {
                'chapter_name': chapter_name,
                'text_blocks': text_blocks,
                'image_blocks': image_blocks,
                'text_path': text_output_path,
                'processed_timestamp': time.time()
            }
            
            # Save to cache
            self._save_to_cache(result)
            
            return result
        except Exception as e:
            logger.error(f"Error processing chapter: {str(e)}")
            raise
        finally:
            self.cleanup()

    def __del__(self):
        """Clean up resources."""
        self.cleanup() 
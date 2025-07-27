# multimodal.py
import io  # This is what was missing!
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

class MultiModalProcessor:
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the multimodal processor
        
        Args:
            tesseract_cmd: Path to tesseract executable (if not in PATH)
        """
        # Configure tesseract if needed (especially for Windows)
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract images and their text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries containing image data and extracted text
        """
        try:
            doc = fitz.open(pdf_path)
            images_data = []
            
            for page_num, page in enumerate(doc):
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            # Convert to PNG bytes
                            img_data = pix.tobytes("png")
                            
                            # Convert to PIL Image using io.BytesIO
                            img_pil = Image.open(io.BytesIO(img_data))
                            
                            # Extract text from image using OCR
                            try:
                                text = pytesseract.image_to_string(img_pil)
                                
                                # Only add if we got meaningful text
                                if text.strip():
                                    images_data.append({
                                        'page': page_num + 1,  # 1-indexed
                                        'image_index': img_index,
                                        'text': text.strip(),
                                        'type': 'image_ocr',
                                        'source': Path(pdf_path).name
                                    })
                                    logger.info(f"Extracted text from image on page {page_num + 1}")
                            
                            except Exception as e:
                                logger.warning(f"OCR failed for image on page {page_num + 1}: {e}")
                        
                        pix = None  # Free memory
                        
                    except Exception as e:
                        logger.error(f"Failed to process image {img_index} on page {page_num + 1}: {e}")
            
            doc.close()
            return images_data
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            return []
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract tables from PDF (requires additional libraries)
        """
        try:
            import camelot  # pip install camelot-py[cv]
            
            tables = camelot.read_pdf(pdf_path, pages='all')
            tables_data = []
            
            for i, table in enumerate(tables):
                # Convert table to string format
                table_text = table.df.to_string()
                
                tables_data.append({
                    'page': table.page,
                    'table_index': i,
                    'text': table_text,
                    'type': 'table',
                    'source': Path(pdf_path).name
                })
            
            return tables_data
            
        except ImportError:
            logger.warning("Camelot not installed. Install with: pip install camelot-py[cv]")
            return []
        except Exception as e:
            logger.error(f"Failed to extract tables: {e}")
            return []
    
    def process_pdf_complete(self, pdf_path: str) -> Dict:
        """
        Extract all content from PDF including text, images, and tables
        """
        results = {
            'text': [],
            'images': [],
            'tables': [],
            'metadata': {
                'source': Path(pdf_path).name,
                'path': str(pdf_path)
            }
        }
        
        # Extract regular text (you already have this)
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    results['text'].append({
                        'page': page_num + 1,
                        'content': text,
                        'type': 'text'
                    })
            doc.close()
        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
        
        # Extract images with OCR
        results['images'] = self.extract_images_from_pdf(pdf_path)
        
        # Extract tables
        results['tables'] = self.extract_tables_from_pdf(pdf_path)
        
        return results
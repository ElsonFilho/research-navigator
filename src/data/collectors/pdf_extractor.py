"""
PDF Text Extractor for Research Navigator
Extracts full text from academic papers with robust error handling
"""

import io
import requests
import time
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

# PDF extraction libraries
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    
try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Robust PDF text extractor with multiple fallback methods
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialize PDF extractor
        
        Args:
            timeout: Timeout for PDF download in seconds
            max_retries: Maximum retry attempts for failed downloads
        """
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Check available libraries
        self.methods = []
        if PDFMINER_AVAILABLE:
            self.methods.append('pdfminer')
        if PYPDF2_AVAILABLE:
            self.methods.append('pypdf2')
            
        if not self.methods:
            raise ImportError(
                "No PDF extraction libraries available. "
                "Install with: pip install PyPDF2 pdfminer.six"
            )
        
        logger.info(f"PDF Extractor initialized with methods: {self.methods}")
    
    
    def download_pdf(self, pdf_url: str) -> Optional[bytes]:
        """
        Download PDF from URL with retries
        
        Args:
            pdf_url: URL to PDF file
            
        Returns:
            PDF content as bytes, or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading PDF (attempt {attempt + 1}/{self.max_retries}): {pdf_url}")
                
                response = requests.get(
                    pdf_url,
                    timeout=self.timeout,
                    headers={'User-Agent': 'ResearchNavigator/1.0'}
                )
                response.raise_for_status()
                
                # Verify it's actually a PDF
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' not in content_type.lower() and not pdf_url.endswith('.pdf'):
                    logger.warning(f"Response may not be PDF: {content_type}")
                
                logger.info(f"Successfully downloaded PDF ({len(response.content)} bytes)")
                return response.content
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout downloading PDF (attempt {attempt + 1})")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error downloading PDF: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        return None
    
    
    def extract_with_pdfminer(self, pdf_content: bytes) -> Tuple[str, int]:
        """
        Extract text using pdfminer.six (most reliable)
        
        Args:
            pdf_content: PDF file as bytes
            
        Returns:
            Tuple of (extracted_text, page_count)
        """
        try:
            pdf_file = io.BytesIO(pdf_content)
            text = pdfminer_extract(pdf_file)
            
            # Estimate page count (rough)
            page_count = len(text) // 2000  # ~2000 chars per page average
            
            logger.info(f"PDFMiner extracted {len(text)} chars (~{page_count} pages)")
            return text.strip(), max(1, page_count)
            
        except Exception as e:
            logger.error(f"PDFMiner extraction failed: {e}")
            raise
    
    
    def extract_with_pypdf2(self, pdf_content: bytes) -> Tuple[str, int]:
        """
        Extract text using PyPDF2 (fallback method)
        
        Args:
            pdf_content: PDF file as bytes
            
        Returns:
            Tuple of (extracted_text, page_count)
        """
        try:
            pdf_file = io.BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            page_count = len(reader.pages)
            text_parts = []
            
            for page_num in range(page_count):
                try:
                    page = reader.pages[page_num]
                    text_parts.append(page.extract_text())
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {e}")
            
            text = "\n".join(text_parts)
            logger.info(f"PyPDF2 extracted {len(text)} chars ({page_count} pages)")
            return text.strip(), page_count
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise
    
    
    def extract_text(self, pdf_content: bytes) -> Tuple[str, int]:
        """
        Extract text using best available method
        
        Args:
            pdf_content: PDF file as bytes
            
        Returns:
            Tuple of (extracted_text, page_count)
        """
        # Try methods in order of reliability
        for method in self.methods:
            try:
                if method == 'pdfminer':
                    return self.extract_with_pdfminer(pdf_content)
                elif method == 'pypdf2':
                    return self.extract_with_pypdf2(pdf_content)
            except Exception as e:
                logger.warning(f"Method {method} failed, trying next: {e}")
                continue
        
        # If all methods failed
        raise Exception("All extraction methods failed")
    
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text (remove excessive whitespace, etc.)
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        
        # Rejoin with single newlines
        cleaned = '\n'.join(lines)
        
        # Replace multiple spaces with single space
        import re
        cleaned = re.sub(r' +', ' ', cleaned)
        
        return cleaned
    
    
    def validate_extraction(self, text: str, min_length: int = 1000) -> bool:
        """
        Validate that extraction succeeded
        
        Args:
            text: Extracted text
            min_length: Minimum acceptable length
            
        Returns:
            True if valid, False otherwise
        """
        if not text or len(text) < min_length:
            logger.warning(f"Extraction too short: {len(text)} chars")
            return False
        
        # Check for reasonable word count
        words = text.split()
        if len(words) < 200:
            logger.warning(f"Too few words: {len(words)}")
            return False
        
        return True
    
    
    def extract_paper(self, paper_data: Dict) -> Dict:
        """
        Extract full text from a paper and add to paper data
        
        Args:
            paper_data: Paper dictionary with 'pdf_url' field
            
        Returns:
            Updated paper dictionary with extraction fields
        """
        pdf_url = paper_data.get('pdf_url')
        if not pdf_url:
            logger.error("No PDF URL in paper data")
            paper_data['extraction_status'] = 'no_url'
            return paper_data
        
        try:
            # Download PDF
            pdf_content = self.download_pdf(pdf_url)
            if not pdf_content:
                paper_data['extraction_status'] = 'download_failed'
                return paper_data
            
            # Extract text
            text, page_count = self.extract_text(pdf_content)
            
            # Clean text
            text = self.clean_text(text)
            
            # Validate
            if not self.validate_extraction(text):
                paper_data['extraction_status'] = 'extraction_failed'
                return paper_data
            
            # Add to paper data
            paper_data['full_text'] = text
            paper_data['full_text_length'] = len(text)
            paper_data['num_pages'] = page_count
            paper_data['extraction_status'] = 'success'
            paper_data['extraction_date'] = datetime.now().isoformat()
            
            logger.info(
                f"✅ Extracted {len(text)} chars from {page_count} pages: "
                f"{paper_data.get('title', 'Unknown')[:50]}..."
            )
            
            return paper_data
            
        except Exception as e:
            logger.error(f"Error extracting paper: {e}")
            paper_data['extraction_status'] = 'error'
            paper_data['extraction_error'] = str(e)
            return paper_data
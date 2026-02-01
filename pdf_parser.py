import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import pandas as pd
import re

# PDF Libraries
import PyPDF2
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF

# Setup logging
temp_log_file_path = '/Users/shubhamagrawal/Documents/MS fall 25/MS fall\'25/ra/data/clinical_pharmacology'
log_file_name = 'pdf_parsing.log'
LOG_FILE_PATH = os.path.join(temp_log_file_path, log_file_name)

""" update the logging logic"""
# if os.path.exists(LOG_FILE_PATH):
#     current_datetime = (datetime.now().date()).replace('-','_')
#     filename = log_file_name.split('.')[0]
    
    
            
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Data class for document sections"""
    title: str
    level: int  # 1 = main section, 2 = subsection, etc.
    content: str
    page_number: int
    start_position: int  # Character position in full text


@dataclass
class PDFParseResult:
    """Data class for parsing results"""
    filename: str
    success: bool
    text_content: str
    sections: List[Dict]  # List of sections found
    metadata: Dict
    page_count: int
    has_images: bool
    is_scanned: bool
    has_redactions: bool
    processing_time: float
    error: Optional[str] = None
    full_file_path: Optional[str] = None


class SectionDetector:
    """Detects sections in PDF text using various heuristics"""
    
    def __init__(self):
        # Common section header patterns
        self.header_patterns = [
            r'^(\d+\.)+\s+([A-Z][^\n]{3,100})$',
            r'^([IVX]+)\.\s+([A-Z][^\n]{3,100})$',
            r'^([A-Z])\.\s+([A-Z][^\n]{3,100})$',
            r'^([A-Z][A-Z\s]{3,50})$',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5}):\s*$',
        ]
        
        # Common section keywords
        self.section_keywords = [
            'abstract', 'introduction', 'background', 'methodology', 'methods',
            'results', 'discussion', 'conclusion', 'references', 'bibliography',
            'summary', 'overview', 'objectives', 'scope', 'definitions',
            'findings', 'recommendations', 'appendix', 'glossary', 'index'
        ]
        
       # Header/Footer and Watermark patterns
        self.header_footer_patterns = [
            r'^\s*page\s+\d+\s*$',
            r'^\s*\d+\s*$',
            r'^\s*\d+\s+of\s+\d+\s*$',
            r'^\s*\d+\s*/\s*\d+\s*$',
            r'^\s*confidential\s*$',
            r'^\s*draft\s*$',
            r'^\s*proprietary\s*$',
            r'^\s*do not distribute\s*$',
            r'^\s*internal use only\s*$',
            r'^\s*appears this way\s*$',
            r'^\s*best possible copy\s*$',
            
        ]
        
        self.watermark_patterns = [
            r'confidential',
            r'draft',
            r'preliminary',
            r'for review only',
            r'not for distribution',
            r'best possible copy',
        ]
       
    
    def detect_repeated_headers_footers(self, pages_text: List[str]) -> Dict[str, List[str]]:
        """Detect repeated headers/footers across pages"""
        if len(pages_text) < 2:
            return {'headers': [], 'footers': []}
        
        first_lines = []
        last_lines = []
        
        for page_text in pages_text:
            lines = [line.strip() for line in page_text.split('\n') if line.strip()]
            if lines:
                first_lines.append(lines[0] if len(lines) > 0 else '')
                last_lines.append(lines[-1] if len(lines) > 0 else '')
        
        from collections import Counter
        header_counter = Counter(first_lines)
        footer_counter = Counter(last_lines)
        
        threshold = len(pages_text) * 0.5
        
        headers = [line for line, count in header_counter.items() 
                  if count >= threshold and len(line) > 0]
        footers = [line for line, count in footer_counter.items() 
                  if count >= threshold and len(line) > 0]
        
        return {'headers': headers, 'footers': footers}
    
    def clean_text_boundaries(self, text: str, pages_text: List[str] = None) -> str:
        """Remove headers, footers, page numbers, and watermarks from text"""
        lines = text.split('\n')
        cleaned_lines = []
        
        repeated = {'headers': [], 'footers': []}
        if pages_text:
            repeated = self.detect_repeated_headers_footers(pages_text)
        
        for line in lines:
            line_stripped = line.strip().lower()
            
            if not line_stripped:
                cleaned_lines.append('')
                continue
            
            is_header_footer = False
            for pattern in self.header_footer_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    is_header_footer = True
                    break
            
            if line.strip() in repeated['headers'] or line.strip() in repeated['footers']:
                is_header_footer = True
            
            is_watermark = False
            for pattern in self.watermark_patterns:
                if re.search(pattern, line_stripped, re.IGNORECASE) and len(line_stripped) < 50:
                    is_watermark = True
                    break
            
            if not is_header_footer and not is_watermark:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_content_boundaries(self, pdf_path: str) -> Dict:
        """Extract content within margins, excluding headers/footers"""
        try:
            doc = fitz.open(pdf_path)
            pages_text = []
            content_boxes = []
            
            for page_num, page in enumerate(doc):
                page_rect = page.rect
                page_height = page_rect.height
                page_width = page_rect.width
                
                # Exclude top 10% and bottom 10% for headers/footers
                top_margin = page_height * 0.10
                bottom_margin = page_height * 0.90
                left_margin = page_width * 0.05
                right_margin = page_width * 0.95
                
                content_box = fitz.Rect(left_margin, top_margin, right_margin, bottom_margin)
                content_boxes.append(content_box)
                
                page_text = page.get_text(clip=content_box)
                pages_text.append(page_text)
            
            doc.close()
            
            full_text = '\n\n'.join(pages_text)
            cleaned_text = self.clean_text_boundaries(full_text, pages_text)
            
            return {
                'cleaned_text': cleaned_text,
                'pages_text': pages_text,
                'content_boxes': content_boxes,
                'boundary_method': 'margin_based_with_pattern_cleaning'
            }
            
        except Exception as e:
            logger.warning(f"Boundary extraction failed: {e}")
            return {
                'cleaned_text': '',
                'pages_text': [],
                'content_boxes': [],
                'boundary_method': 'failed'
            }
    
    def detect_sections_from_text(self, text: str) -> List[Section]:
        """Detect sections from plain text using patterns"""
        sections = []
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped:
                if current_content:
                    current_content.append('')
                continue
            
            is_header = False
            header_level = 1
            header_title = None
            
            for pattern in self.header_patterns:
                match = re.match(pattern, line_stripped)
                if match:
                    is_header = True
                    if '.' in match.group(1):
                        header_level = match.group(1).count('.')
                        header_title = match.group(2) if len(match.groups()) > 1 else line_stripped
                    else:
                        header_title = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    break
            
            if not is_header:
                line_lower = line_stripped.lower()
                for keyword in self.section_keywords:
                    if keyword in line_lower and len(line_stripped) < 100:
                        if (line_stripped.isupper() or 
                            line_stripped[0].isupper() and len(line_stripped.split()) < 10):
                            is_header = True
                            header_title = line_stripped
                            header_level = 1
                            break
            
            if is_header and header_title:
                if current_section:
                    current_section['content'] = '\n'.join(current_content).strip()
                    sections.append(current_section)
                
                current_section = {
                    'title': header_title.strip(),
                    'level': header_level,
                    'content': '',
                    'page_number': -1,
                    'start_position': len('\n'.join(lines[:i]))
                }
                current_content = []
            else:
                if current_content or current_section:
                    current_content.append(line)
        
        if current_section:
            current_section['content'] = '\n'.join(current_content).strip()
            sections.append(current_section)
        
        return sections
    
    def detect_sections_from_structure(self, pdf_path: str) -> List[Section]:
        """Detect sections using PDF structure (fonts, sizes, styles)"""
        sections = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            size = span["size"]
                            flags = span["flags"]
                            
                            is_bold = flags & 2 ** 4
                            
                            if (is_bold or size > 12) and len(text) > 3 and len(text) < 100:
                                if (text[0].isupper() and 
                                    any(keyword in text.lower() for keyword in self.section_keywords)):
                                    
                                    if size > 16:
                                        level = 1
                                    elif size > 14:
                                        level = 2
                                    else:
                                        level = 3
                                    
                                    sections.append({
                                        'title': text,
                                        'level': level,
                                        'content': '',
                                        'page_number': page_num + 1,
                                        'start_position': -1
                                    })
            
            doc.close()
            
        except Exception as e:
            logger.debug(f"Structure-based section detection failed: {e}")
        
        return sections
    
    def merge_and_extract_content(
        self, 
        text_sections: List[Dict], 
        structure_sections: List[Dict],
        full_text: str
    ) -> List[Dict]:
        """Merge sections from text and structure analysis, extract content"""
        
        if structure_sections and len(structure_sections) > 2:
            sections = structure_sections
        elif text_sections:
            sections = text_sections
        else:
            return [{
                'title': 'Document Content',
                'level': 1,
                'content': full_text,
                'page_number': 1,
                'start_position': 0,
                'content_length': len(full_text)
            }]
        
        for i, section in enumerate(sections):
            if i < len(sections) - 1:
                next_section = sections[i + 1]
                
                title_pos = full_text.find(section['title'])
                next_title_pos = full_text.find(next_section['title'], title_pos + 1)
                
                if title_pos != -1 and next_title_pos != -1:
                    content_start = title_pos + len(section['title'])
                    raw_content = full_text[content_start:next_title_pos]
                    
                    lines = raw_content.split('\n')
                    cleaned_lines = [line for line in lines if line.strip() != section['title'].strip()]
                    
                    section['content'] = '\n'.join(cleaned_lines).strip()
                    section['start_position'] = title_pos
                    section['content_length'] = len(section['content'])
                else:
                    section['content'] = ""
                    section['content_length'] = 0
            else:
                title_pos = full_text.find(section['title'])
                if title_pos != -1:
                    content_start = title_pos + len(section['title'])
                    raw_content = full_text[content_start:]
                    
                    lines = raw_content.split('\n')
                    cleaned_lines = [line for line in lines if line.strip() != section['title'].strip()]
                    
                    section['content'] = '\n'.join(cleaned_lines).strip()
                    section['start_position'] = title_pos
                    section['content_length'] = len(section['content'])
                else:
                    section['content'] = ""
                    section['content_length'] = 0
        
        return sections


class PDFParser:
    """Multi-strategy PDF parser with section and boundary detection"""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.section_detector = SectionDetector()
    
    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """Quick check if PDF is scanned"""
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                text = page.get_text().strip()
                if len(text) < 50 and len(page.get_images()) > 0:
                    doc.close()
                    return True
            doc.close()
            return False
        except Exception:
            return False
    
    def detect_redactions(self, pdf_path: str) -> bool:
        """Detect if PDF has redacted sections"""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                annots = page.annots()
                if annots:
                    for annot in annots:
                        if annot.type[0] == 12:
                            doc.close()
                            return True
                
                drawings = page.get_drawings()
                for drawing in drawings:
                    if drawing.get('fill', None) == (0, 0, 0):
                        doc.close()
                        return True
            doc.close()
            return False
        except Exception:
            return False
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            text = []
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            logger.debug(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber"""
        try:
            text = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return '\n'.join(text)
        except Exception as e:
            logger.debug(f"pdfplumber extraction failed: {e}")
            return ""
    
    def extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = []
            for page in doc:
                text.append(page.get_text())
            doc.close()
            return '\n'.join(text)
        except Exception as e:
            logger.debug(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def extract_text_with_boundaries(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text with boundary detection (removes headers/footers)
        Returns: (cleaned_text, boundary_info)
        """
        try:
            boundary_result = self.section_detector.extract_content_boundaries(pdf_path)
            
            if boundary_result['cleaned_text']:
                logger.info(f"✓ Boundary detection applied: {os.path.basename(pdf_path)}")
                return boundary_result['cleaned_text'], {
                    'boundary_applied': True,
                    'method': boundary_result['boundary_method'],
                    'pages_processed': len(boundary_result['pages_text'])
                }
            else:
                logger.debug(f"Boundary detection failed, using regular extraction")
                return self.extract_text_pymupdf(pdf_path), {
                    'boundary_applied': False,
                    'method': 'regular_extraction',
                    'pages_processed': 0
                }
                
        except Exception as e:
            logger.debug(f"Boundary extraction failed: {e}")
            return self.extract_text_pymupdf(pdf_path), {
                'boundary_applied': False,
                'method': 'fallback',
                'pages_processed': 0
            }
    
    def extract_text_ocr(self, pdf_path: str, dpi: int = 200) -> str:
        """Extract text from scanned PDF using OCR"""
        try:
            images = convert_from_path(
                pdf_path, 
                dpi=dpi,
                thread_count=2,
                grayscale=True
            )
            
            text = []
            for i, image in enumerate(images):
                custom_config = r'--oem 1 --psm 6'
                page_text = pytesseract.image_to_string(image, config=custom_config)
                text.append(page_text)
                
                
            
            return '\n'.join(text)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def extract_metadata(self, pdf_path: str) -> Dict:
        """Extract PDF metadata"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            page_count = len(doc)
            
            has_images = False
            for page in doc:
                if len(page.get_images()) > 0:
                    has_images = True
                    break
            
            doc.close()
            
            return {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'page_count': page_count,
                'has_images': has_images
            }
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}
    
    def parse_pdf(self, pdf_path: str) -> PDFParseResult:
        """Main parsing function with section and boundary detection"""
        start_time = datetime.now()
        filename = os.path.basename(pdf_path)
        
        try:
            metadata = self.extract_metadata(pdf_path)
            page_count = metadata.get('page_count', 0)
            has_images = metadata.get('has_images', False)
            
            is_scanned = self.is_scanned_pdf(pdf_path)
            has_redactions = self.detect_redactions(pdf_path)
            
            text_content = ""
            boundary_info = {}
            
            # Extract text with boundary detection
            if is_scanned:
                logger.info(f"Scanned PDF: {filename}, using OCR")
                text_content = self.extract_text_ocr(pdf_path)
                boundary_info = {'boundary_applied': False, 'method': 'ocr', 'pages_processed': 0}
            else:
                # add extract_text_pymupdf usage here as primary maybe?
                text_content, boundary_info = self.extract_text_with_boundaries(pdf_path)
                
                if len(text_content.strip()) < 100:
                    logger.info(f"Boundary extraction too short, trying pdfplumber")
                    text_content = self.extract_text_pdfplumber(pdf_path)
                    boundary_info['boundary_applied'] = False
                
                if len(text_content.strip()) < 100:
                    text_content = self.extract_text_pypdf2(pdf_path)
                    boundary_info['boundary_applied'] = False
                
                if len(text_content.strip()) < 100 and has_images:
                    logger.info(f"Fallback to OCR for {filename}")
                    text_content = self.extract_text_ocr(pdf_path)
                    boundary_info['boundary_applied'] = False
            
            # Detect sections
            sections = []
            if text_content:
                logger.info(f"Detecting sections in {filename}")
                
                structure_sections = self.section_detector.detect_sections_from_structure(pdf_path)
                text_sections = self.section_detector.detect_sections_from_text(text_content)
                
                sections = self.section_detector.merge_and_extract_content(
                    text_sections, 
                    structure_sections, 
                    text_content
                )
                
                logger.info(f"Found {len(sections)} sections in {filename}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return PDFParseResult(
                filename=filename,
                success=True,
                text_content=text_content,
                sections=sections,
                metadata={**metadata, **boundary_info},
                page_count=page_count,
                has_images=has_images,
                is_scanned=is_scanned,
                has_redactions=has_redactions,
                processing_time=processing_time,
                full_file_path=pdf_path
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to parse {filename}: {e}")
            return PDFParseResult(
                filename=filename,
                success=False,
                text_content="",
                sections=[],
                metadata={},
                page_count=0,
                has_images=False,
                is_scanned=False,
                has_redactions=False,
                processing_time=processing_time,
                error=str(e),
                full_file_path=pdf_path
            )


def save_individual_json(result: PDFParseResult, output_dir: str) -> str:
    """Save individual JSON file for each PDF with sections"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = Path(result.filename).stem
        json_filename = f"{base_name}_parsed.json"
        json_path = os.path.join(output_dir, json_filename)
        
        formatted_sections = []
        for section in result.sections:
            formatted_sections.append({
                'section_title': section.get('title', ''),
                'section_level': section.get('level', 1),
                'section_content': section.get('content', ''),
                'content_length': section.get('content_length', len(section.get('content', ''))),
                'page_number': section.get('page_number', -1),
                'start_position': section.get('start_position', -1)
            })
        
        boundary_info = {
            'boundary_detection_applied': result.metadata.get('boundary_applied', False),
            'boundary_method': result.metadata.get('method', 'none'),
            'pages_with_boundaries': result.metadata.get('pages_processed', 0)
        }
        
        json_data = {
            'filename': result.filename,
            'full_path': result.full_file_path,
            'success': result.success,
            'parsed_timestamp': datetime.now().isoformat(),
            
            'full_text_content': result.text_content,
            'total_text_length': len(result.text_content),
            
            'sections': formatted_sections,
            'total_sections': len(result.sections),
            
            'metadata': {
                'title': result.metadata.get('title', ''),
                'author': result.metadata.get('author', ''),
                'subject': result.metadata.get('subject', ''),
                'creator': result.metadata.get('creator', ''),
                'producer': result.metadata.get('producer', ''),
                'creation_date': result.metadata.get('creation_date', ''),
                'page_count': result.metadata.get('page_count', 0),
                'has_images': result.metadata.get('has_images', False)
            },
            
            'boundary_detection': boundary_info,
            
            'page_count': result.page_count,
            'has_images': result.has_images,
            'is_scanned': result.is_scanned,
            'has_redactions': result.has_redactions,
            'processing_time_seconds': result.processing_time,
            'error': result.error
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return json_path
        
    except Exception as e:
        logger.error(f"Failed to save JSON for {result.filename}: {e}")
        return None


def process_single_pdf_with_tracking(args):
    """Worker function for multiprocessing"""
    pdf_path, tesseract_path, output_dir = args
    
    parser = PDFParser(tesseract_path)
    result = parser.parse_pdf(pdf_path)
    
    json_path = save_individual_json(result, output_dir)
    
    return {
        'filename': result.filename,
        'full_path': pdf_path,
        'parsed': result.success,
        'json_output_path': json_path,
        'page_count': result.page_count,
        'text_length': len(result.text_content),
        'section_count': len(result.sections),
        'is_scanned': result.is_scanned,
        'has_redactions': result.has_redactions,
        'has_images': result.has_images,
        'processing_time': result.processing_time,
        'parsed_timestamp': datetime.now().isoformat(),
        'error_message': result.error,
        'success': result.success
    }


def batch_process_with_dataframe(
    df: pd.DataFrame,
    pdf_path_column: str,
    output_json_dir: str,
    output_df_path: str,
    num_workers: Optional[int] = None,
    tesseract_path: Optional[str] = None,
    resume: bool = True
):
    """
    Process PDFs with DataFrame tracking, section detection, and boundary detection
    """
    
    if pdf_path_column not in df.columns:
        raise ValueError(f"Column '{pdf_path_column}' not found in DataFrame")
    
    tracking_columns = {
        'parsed': False,
        'json_output_path': None,
        'page_count': 0,
        'text_length': 0,
        'section_count': 0,
        'is_scanned': False,
        'has_redactions': False,
        'has_images': False,
        'processing_time': 0.0,
        'parsed_timestamp': None,
        'error_message': None
    }
    
    for col, default_val in tracking_columns.items():
        if col not in df.columns:
            df[col] = default_val
    
    if resume:
        df_to_process = df[df['parsed'] == False].copy()
        logger.info(f"Resume mode: Found {len(df_to_process)} unparsed files")
    else:
        df_to_process = df.copy()
    
    total_files = len(df_to_process)
    
    if total_files == 0:
        logger.info("All files already parsed!")
        return df
    
    logger.info(f"Processing {total_files} PDF files")
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    logger.info(f"Using {num_workers} workers")
    
    worker_args = [
        (row[pdf_path_column], tesseract_path, output_json_dir)
        for _, row in df_to_process.iterrows()
    ]
    
    completed = 0
    successful = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_pdf_with_tracking, arg): arg[0]
            for arg in worker_args
        }
        
        for future in as_completed(futures):
            result_data = future.result()
            completed += 1
            
            # Update DataFrame
            mask = df[pdf_path_column] == result_data['full_path']
            for key, value in result_data.items():
                if key in df.columns:
                    df.loc[mask, key] = value
            
            if result_data['success']:
                successful += 1
            
            # Save periodically
            if completed % 100 == 0:
                df.to_csv(output_df_path, index=False)
                logger.info(f"Progress: {completed}/{total_files} "
                           f"({completed/total_files*100:.1f}%) | "
                           f"Success: {successful} | "
                           f"Failed: {completed - successful}")
    
    # Final save
    df.to_csv(output_df_path, index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total processed: {completed}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {completed - successful}")
    logger.info(f"Updated DataFrame saved to: {output_df_path}")
    logger.info(f"Individual JSONs saved to: {output_json_dir}")
    logger.info(f"{'='*60}\n")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parse PDFs with section detection and DataFrame tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_csv', help='Input CSV file containing PDF paths')
    parser.add_argument('path_column', help='Column name containing PDF file paths')
    parser.add_argument('output_json_dir', help='Directory to save individual JSON files')
    parser.add_argument('output_csv', help='Output CSV file with tracking data')
    parser.add_argument('--workers', type=int, help='Number of workers')
    parser.add_argument('--tesseract', help='Path to tesseract executable')
    parser.add_argument('--no-resume', action='store_true', help='Reprocess all files')
    
    args = parser.parse_args()
    
    logger.info(f"Loading DataFrame from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    logger.info(f"Loaded {len(df)} records")
    
    updated_df = batch_process_with_dataframe(
        df=df,
        pdf_path_column=args.path_column,
        output_json_dir=args.output_json_dir,
        output_df_path=args.output_csv,
        num_workers=args.workers,
        tesseract_path=args.tesseract,
        resume=not args.no_resume
    )
    
    print(f"\nProcessing complete! Updated DataFrame saved to: {args.output_csv}")
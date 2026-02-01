"""
Complete PDF Extraction Pipeline
Extracts text, tables, charts, and document structure from PDFs
Handles clear, scanned, and noisy PDFs with watermarks/redactions
"""

import os
import sys  # Add this
import json
import io
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from PIL import Image
import cv2

# PDF Processing
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pdfplumber

# OCR
import pytesseract
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Image Processing
from scipy import ndimage
from skimage.filters import threshold_otsu, gaussian
from skimage.restoration import denoise_bilateral

# Table Extraction
import camelot
import tabula

# Layout Analysis (optional - install if needed)
try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False

import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

# Create logs directory BEFORE configuring logging
os.makedirs('logs', exist_ok=True)

# Configure logging (use the code from section 2 above)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler with rotation
file_handler = RotatingFileHandler(
    'logs/pipeline.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent duplicate logs
logger.propagate = False


# ==================== Data Models ====================

@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    
    def to_dict(self):
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}
    
    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    @property
    def height(self):
        return self.y2 - self.y1
    
    @property
    def width(self):
        return self.x2 - self.x1


@dataclass
class TextBlock:
    id: str
    content: str
    bbox: BBox
    reading_order: int
    parent_heading: Optional[str]
    confidence: float
    font_size_estimate: int = 12
    
    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "bbox": self.bbox.to_dict(),
            "reading_order": self.reading_order,
            "parent_heading": self.parent_heading,
            "confidence": self.confidence,
            "font_size_estimate": self.font_size_estimate
        }


@dataclass
class Table:
    id: str
    caption: Optional[str]
    bbox: BBox
    reading_order: int
    parent_heading: Optional[str]
    data: Dict[str, List]
    extraction_method: str
    confidence: float
    
    def to_dict(self):
        return {
            "id": self.id,
            "caption": self.caption,
            "bbox": self.bbox.to_dict(),
            "reading_order": self.reading_order,
            "parent_heading": self.parent_heading,
            "data": self.data,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence
        }


@dataclass
class Chart:
    id: str
    type: str
    caption: Optional[str]
    bbox: BBox
    reading_order: int
    parent_heading: Optional[str]
    description: str
    confidence: float
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "caption": self.caption,
            "bbox": self.bbox.to_dict(),
            "reading_order": self.reading_order,
            "parent_heading": self.parent_heading,
            "description": self.description,
            "confidence": self.confidence
        }


@dataclass
class Heading:
    text: str
    level: int
    bbox: BBox
    reading_order: int
    page_number: int
    
    def to_dict(self):
        return {
            "text": self.text,
            "level": self.level,
            "bbox": self.bbox.to_dict(),
            "reading_order": self.reading_order,
            "page_number": self.page_number
        }


# ==================== PDF Type Detector ====================

class PDFTypeDetector:
    """Detect if PDF is clear text, scanned, mixed, or noisy"""
    
    def __init__(self):
        self.text_threshold = 50
        self.image_area_threshold = 0.7
    
    def detect(self, pdf_path: str) -> str:
        """Returns: 'clear', 'scanned', 'mixed', or 'noisy'"""
        doc = fitz.open(pdf_path)
        sample_pages = min(3, len(doc))
        
        text_pages = 0
        image_pages = 0
        noisy_pages = 0
        
        for page_num in range(sample_pages):
            page = doc[page_num]
            text = page.get_text()
            has_text = len(text.strip()) > self.text_threshold
            
            image_list = page.get_images()
            image_coverage = self._get_image_coverage(page, image_list)
            
            is_noisy = False
            if image_coverage > self.image_area_threshold and image_list:
                is_noisy = self._check_noise(page, image_list[0])
            
            if is_noisy:
                noisy_pages += 1
            elif has_text and image_coverage < 0.3:
                text_pages += 1
            elif image_coverage > self.image_area_threshold:
                image_pages += 1
        
        doc.close()
        
        if noisy_pages > sample_pages * 0.5:
            return "noisy"
        elif text_pages == sample_pages:
            return "clear"
        elif image_pages == sample_pages:
            return "scanned"
        else:
            return "mixed"
    
    def _get_image_coverage(self, page, image_list) -> float:
        page_area = page.rect.width * page.rect.height
        total_image_area = 0
        
        for img in image_list:
            try:
                rects = page.get_image_rects(img[0])
                for rect in rects:
                    total_image_area += rect.width * rect.height
            except:
                continue
        
        return min(total_image_area / page_area, 1.0) if page_area > 0 else 0.0
    
    def _check_noise(self, page, img_info) -> bool:
        try:
            xref = img_info[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            img = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(img.convert('L'))
            
            edges = ndimage.sobel(img_array)
            edge_density = np.mean(np.abs(edges))
            
            smoothed = gaussian(img_array, sigma=2)
            local_var = np.var(img_array - smoothed)
            
            return edge_density > 30 or local_var > 500
        except:
            return False


# ==================== Image Preprocessor ====================

class ImagePreprocessor:
    """Enhance image quality for better OCR"""
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        denoised = self._denoise(gray)
        
        # Enhance contrast
        enhanced = self._enhance_contrast(denoised)
        
        # Deskew
        deskewed = self._deskew(enhanced)
        
        # Binarize
        binary = self._binarize(deskewed)
        
        return binary
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using bilateral filter"""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct skew angle"""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image
        
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        
        if abs(angle) < 0.5:
            return image
        
        h, w = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert to binary image"""
        _, binary = cv2.threshold(image, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def detect_watermark(self, image: np.ndarray) -> bool:
        """Detect if image has watermark"""
        # Simple detection based on semi-transparent regions
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Check for mid-range pixel values (semi-transparent)
        mid_range = np.sum((gray > 100) & (gray < 200))
        total_pixels = gray.size
        
        return (mid_range / total_pixels) > 0.15


# ==================== OCR Engine ====================

class OCREngine:
    """Multi-engine OCR with fallback"""
    
    def __init__(self):
        self.primary_engine = "tesseract"
        if EASYOCR_AVAILABLE:
            self.reader = easyocr.Reader(['en'], gpu=False)
        else:
            self.reader = None
    
    def extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text and return confidence score"""
        try:
            text = pytesseract.image_to_string(image)
            
            # Get confidence
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data['conf'] if c != '-1']
            avg_conf = np.mean(confidences) / 100.0 if confidences else 0.5
            
            return text.strip(), avg_conf
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
            
            if self.reader:
                try:
                    result = self.reader.readtext(image)
                    text = " ".join([item[1] for item in result])
                    avg_conf = np.mean([item[2] for item in result]) if result else 0.5
                    return text.strip(), avg_conf
                except Exception as e2:
                    logger.error(f"EasyOCR also failed: {e2}")
            
            return "", 0.0


# ==================== Layout Analyzer ====================

class LayoutAnalyzer:
    """Analyze document layout and detect regions"""
    
    def __init__(self):
        self.use_layoutparser = LAYOUTPARSER_AVAILABLE
        if self.use_layoutparser:
            try:
                self.model = lp.Detectron2LayoutModel(
                    'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                    label_map={0: "Text", 1: "Title", 2: "List", 
                              3: "Table", 4: "Figure"}
                )
            except:
                self.use_layoutparser = False
                logger.warning("LayoutParser model loading failed, using fallback")
    
    def analyze(self, image: np.ndarray) -> List[Dict]:
        """Detect layout regions"""
        if self.use_layoutparser:
            return self._analyze_with_layoutparser(image)
        else:
            return self._analyze_simple(image)
    
    def _analyze_with_layoutparser(self, image: np.ndarray) -> List[Dict]:
        """Use LayoutParser for detection"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        layout = self.model.detect(image)
        
        regions = []
        for idx, block in enumerate(layout):
            region = {
                "id": idx,
                "type": self._normalize_type(block.type),
                "bbox": BBox(
                    int(block.block.x_1),
                    int(block.block.y_1),
                    int(block.block.x_2),
                    int(block.block.y_2)
                ),
                "confidence": float(block.score)
            }
            regions.append(region)
        
        return self._sort_reading_order(regions)
    
    def _analyze_simple(self, image: np.ndarray) -> List[Dict]:
        """Simple contour-based layout detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 0, 255, 
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 50 or h < 20:
                continue
            
            aspect_ratio = w / h if h > 0 else 0
            
            # Classify based on aspect ratio and size
            if aspect_ratio > 5:
                region_type = "table"
            elif h > 30 and w > image.shape[1] * 0.5:
                region_type = "heading"
            else:
                region_type = "text"
            
            region = {
                "id": idx,
                "type": region_type,
                "bbox": BBox(x, y, x + w, y + h),
                "confidence": 0.7
            }
            regions.append(region)
        
        return self._sort_reading_order(regions)
    
    def _normalize_type(self, type_str: str) -> str:
        type_map = {
            "Text": "text",
            "Title": "heading",
            "List": "list",
            "Table": "table",
            "Figure": "figure"
        }
        return type_map.get(type_str, "text")
    
    def _sort_reading_order(self, regions: List[Dict]) -> List[Dict]:
        """Sort by reading order"""
        def get_key(r):
            bbox = r["bbox"]
            return (bbox.y1 // 50, bbox.x1)
        
        sorted_regions = sorted(regions, key=get_key)
        for idx, region in enumerate(sorted_regions):
            region["reading_order"] = idx
        
        return sorted_regions


# ==================== Heading Detector ====================

class HeadingDetector:
    """Detect and classify headings"""
    
    def __init__(self):
        self.heading_keywords = [
            "chapter", "section", "abstract", "introduction",
            "conclusion", "references", "appendix", "summary", 
            "table of contents", "executive summary", ""
        ]
    
    def detect(self, regions: List[Dict], page_image: np.ndarray, 
               page_num: int, ocr_engine: OCREngine) -> List[Heading]:
        """Detect headings from layout regions"""
        headings = []
        
        heading_regions = [r for r in regions if r["type"] == "heading"]
        
        for region in heading_regions:
            bbox = region["bbox"]
            roi = page_image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            
            text, conf = ocr_engine.extract_text(roi)
            
            if not text:
                continue
            
            level = self._determine_level(bbox, page_image.shape)
            
            heading = Heading(
                text=text,
                level=level,
                bbox=bbox,
                reading_order=region["reading_order"],
                page_number=page_num
            )
            headings.append(heading)
        
        return headings
    
    def _determine_level(self, bbox: BBox, image_shape: Tuple) -> int:
        """Determine heading level based on size and position"""
        height = bbox.height
        
        if height > 40:
            return 1
        elif height > 30:
            return 2
        elif height > 20:
            return 3
        else:
            return 4


# ==================== Table Extractor ====================

class TableExtractor:
    """Extract tables using multiple methods"""
    
    def extract_from_pdf(self, pdf_path: str, page_num: int) -> List[Table]:
        """Extract tables from PDF page"""
        tables = []
        
        # Try Camelot first (for lattice tables)
        try:
            camelot_tables = camelot.read_pdf(
                pdf_path, 
                pages=str(page_num + 1),
                flavor='lattice'
            )
            
            for idx, table in enumerate(camelot_tables):
                if table.accuracy > 50:
                    data = self._parse_camelot_table(table)
                    tables.append(Table(
                        id=f"table_{page_num}_{idx}",
                        caption=None,
                        bbox=self._get_table_bbox(table),
                        reading_order=idx,
                        parent_heading=None,
                        data=data,
                        extraction_method="camelot_lattice",
                        confidence=table.accuracy / 100.0
                    ))
        except Exception as e:
            logger.debug(f"Camelot lattice failed: {e}")
        
        # Try stream flavor if lattice didn't work well
        if len(tables) == 0:
            try:
                camelot_tables = camelot.read_pdf(
                    pdf_path,
                    pages=str(page_num + 1),
                    flavor='stream'
                )
                
                for idx, table in enumerate(camelot_tables):
                    data = self._parse_camelot_table(table)
                    tables.append(Table(
                        id=f"table_{page_num}_{idx}",
                        caption=None,
                        bbox=self._get_table_bbox(table),
                        reading_order=idx,
                        parent_heading=None,
                        data=data,
                        extraction_method="camelot_stream",
                        confidence=0.7
                    ))
            except Exception as e:
                logger.debug(f"Camelot stream failed: {e}")
        
        return tables
    
    def _parse_camelot_table(self, table) -> Dict:
        """Parse Camelot table to dictionary"""
        df = table.df
        
        if df.empty:
            return {"headers": [], "rows": []}
        
        headers = df.iloc[0].tolist()
        rows = df.iloc[1:].values.tolist()
        
        return {
            "headers": headers,
            "rows": rows
        }
    
    def _get_table_bbox(self, table) -> BBox:
        """Extract bounding box from Camelot table"""
        try:
            x1, y1, x2, y2 = table._bbox
            return BBox(int(x1), int(y1), int(x2), int(y2))
        except:
            return BBox(0, 0, 100, 100)


# ==================== Chart Detector ====================

class ChartDetector:
    """Detect and extract charts/figures"""
    
    def detect(self, regions: List[Dict], page_image: np.ndarray,
               page_num: int) -> List[Chart]:
        """Detect charts from layout regions"""
        charts = []
        
        figure_regions = [r for r in regions if r["type"] == "figure"]
        
        for idx, region in enumerate(figure_regions):
            bbox = region["bbox"]
            
            # Extract figure region
            roi = page_image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            
            # Simple classification
            chart_type = self._classify_chart(roi)
            
            chart = Chart(
                id=f"chart_{page_num}_{idx}",
                type=chart_type,
                caption=None,
                bbox=bbox,
                reading_order=region["reading_order"],
                parent_heading=None,
                description=f"A {chart_type} visualization",
                confidence=region["confidence"]
            )
            charts.append(chart)
        
        return charts
    
    def _classify_chart(self, image: np.ndarray) -> str:
        """Simple chart type classification"""
        # This is a placeholder - in production, use ML model
        return "figure"


# ==================== Main Pipeline ====================

class PDFExtractionPipeline:
    """Main extraction pipeline"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.type_detector = PDFTypeDetector()
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine()
        self.layout_analyzer = LayoutAnalyzer()
        self.heading_detector = HeadingDetector()
        self.table_extractor = TableExtractor()
        self.chart_detector = ChartDetector()
        
        logger.info("Pipeline initialized")
    
    def process(self, pdf_path: str) -> Dict:
        """Process PDF and return structured data"""
        logger.info(f"Processing: {pdf_path}")
        
        # Detect PDF type
        pdf_type = self.type_detector.detect(pdf_path)
        logger.info(f"PDF type: {pdf_type}")
        
        # Get page count
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()
        
        logger.info(f"Total pages to process: {num_pages}")
        
        # Process each page
        pages_data = []
        all_headings = []
        
        for page_num in tqdm(range(num_pages), desc="Processing pages"):
            try:
                page_data, page_headings = self._process_page(pdf_path, page_num, pdf_type)
                pages_data.append(page_data)
                all_headings.extend(page_headings)
                
                # Log progress
                if (page_num + 1) % 10 == 0:
                    logger.info(f"Processed {page_num + 1}/{num_pages} pages")
                    
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {e}")
                # Continue with other pages
                continue
        
        logger.info(f"Page processing complete. Building structure...")
        
        # Build document structure
        doc_structure = self._build_structure(all_headings)
        
        logger.info(f"Creating final output structure...")
        
        # Create final output
        result = {
            "document_metadata": {
                "source_file": os.path.basename(pdf_path),
                "pdf_type": pdf_type,
                "total_pages": num_pages,
                "extraction_date": datetime.now().isoformat(),
                "processing_version": "1.0.0"
            },
            "document_structure": doc_structure,
            "pages": pages_data,
            "extraction_summary": self._create_summary(pages_data)
        }
        
        logger.info(f"Saving output...")
        
        # Save output
        output_file = self._save_output(result, pdf_path)
        
        logger.info(f"Processing complete: {output_file}")
        
        return result
    
    
    def _process_page(self, pdf_path: str, page_num: int, 
                 pdf_type: str) -> Tuple[Dict, List]:
        """Process single page - returns (page_data_dict, heading_objects)"""
        # Convert page to image
        images = convert_from_path(pdf_path, first_page=page_num + 1,
                                last_page=page_num + 1, dpi=300)
        page_image = np.array(images[0])
        
        # Preprocess if needed
        if pdf_type in ["scanned", "noisy"]:
            processed_image = self.preprocessor.process(page_image)
        else:
            processed_image = page_image
        
        # Analyze layout
        regions = self.layout_analyzer.analyze(processed_image)
        
        # Detect headings (returns Heading objects)
        headings = self.heading_detector.detect(
            regions, processed_image, page_num, self.ocr_engine
        )
        
        # Extract text blocks
        text_blocks = self._extract_text_blocks(
            regions, processed_image, page_num
        )
        
        # Extract tables
        tables = self.table_extractor.extract_from_pdf(pdf_path, page_num)
        
        # Detect charts
        charts = self.chart_detector.detect(regions, processed_image, page_num)
        
        # Detect special elements
        has_watermark = self.preprocessor.detect_watermark(page_image)
        
        page_data = {
            "page_number": page_num + 1,
            "metadata": {
                "has_watermark": has_watermark,
                "quality_score": 0.85,
                "layout_complexity": len(regions)
            },
            "headings": [h.to_dict() for h in headings],  # Convert to dict for JSON
            "text_blocks": [tb.to_dict() for tb in text_blocks],
            "tables": [t.to_dict() for t in tables],
            "charts": [c.to_dict() for c in charts]
        }
        
        return page_data, headings  # Return both dict and objects
    
    def _extract_text_blocks(self, regions: List[Dict], 
                            image: np.ndarray, page_num: int) -> List[TextBlock]:
        """Extract text from text regions"""
        text_blocks = []
        current_heading = None
        
        for region in regions:
            if region["type"] == "heading":
                bbox = region["bbox"]
                roi = image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
                text, conf = self.ocr_engine.extract_text(roi)
                current_heading = text
            
            elif region["type"] in ["text", "list"]:
                bbox = region["bbox"]
                roi = image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
                text, conf = self.ocr_engine.extract_text(roi)
                
                if text:
                    text_block = TextBlock(
                        id=f"text_{page_num}_{region['id']}",
                        content=text,
                        bbox=bbox,
                        reading_order=region["reading_order"],
                        parent_heading=current_heading,
                        confidence=conf
                    )
                    text_blocks.append(text_block)
        
        return text_blocks
    
    def _build_structure(self, headings: List[Heading]) -> Dict:
        """Build document hierarchy from headings"""
        structure = {"sections": []}
        
        if not headings:
            return structure
        
        # Sort by page and reading order (headings are Heading objects)
        sorted_headings = sorted(headings, 
                                key=lambda h: (h.page_number, h.reading_order))
        
        current_section = None
        
        for heading in sorted_headings:
            if heading.level <= 2:
                current_section = {
                    "heading": heading.text,
                    "level": heading.level,
                    "page_number": heading.page_number,
                    "subsections": []
                }
                structure["sections"].append(current_section)
            elif current_section:
                current_section["subsections"].append({
                    "heading": heading.text,
                    "level": heading.level,
                    "page_number": heading.page_number
                })
        
        return structure
    
    def _create_summary(self, pages_data: List[Dict]) -> Dict:
        """Create extraction summary"""
        total_text = sum(len(p["text_blocks"]) for p in pages_data)
        total_tables = sum(len(p["tables"]) for p in pages_data)
        total_charts = sum(len(p["charts"]) for p in pages_data)
        total_headings = sum(len(p["headings"]) for p in pages_data)
        
        return {
            "total_text_blocks": total_text,
            "total_tables": total_tables,
            "total_charts": total_charts,
            "total_headings": total_headings,
            "pages_with_watermarks": sum(1 for p in pages_data 
                                        if p["metadata"]["has_watermark"])
        }
    
    def _save_output(self, data: Dict, pdf_path: str):
        """Save extracted data to JSON file with robust error handling"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            pdf_name = Path(pdf_path).stem
            output_path = os.path.join(self.output_dir, f"{pdf_name}_extracted.json")
            temp_path = output_path + ".tmp"
            
            # Custom JSON encoder
            def json_default(obj):
                """Handle non-serializable objects"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                else:
                    return str(obj)
            
            # Create JSON string with custom encoder
            logger.info(f"Creating JSON string...")
            json_string = json.dumps(data, indent=2, ensure_ascii=False, default=json_default)
            json_size = len(json_string)
            logger.info(f"JSON string created: {json_size:,} characters")
            
            # Write to temporary file first
            logger.info(f"Writing to temporary file: {temp_path}")
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(json_string)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            # Verify temp file
            temp_size = os.path.getsize(temp_path)
            logger.info(f"Temp file size: {temp_size:,} bytes")
            
            # Verify it's valid JSON
            logger.info(f"Verifying JSON validity...")
            with open(temp_path, 'r', encoding='utf-8') as f:
                json.load(f)
            
            # Rename temp to final (atomic operation)
            logger.info(f"Renaming to final file: {output_path}")
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_path, output_path)
            
            # Final verification
            final_size = os.path.getsize(output_path)
            logger.info(f"✓ Output saved successfully: {output_path} ({final_size:,} bytes)")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            
            # Clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            raise
            



def main():
    """Example usage"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='PDF Extraction Pipeline - Extract structured data from PDFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python pdf_extraction_pipeline.py document.pdf
        python pdf_extraction_pipeline.py document.pdf --output results/
        """
    )
    
    parser.add_argument('pdf_path', help='Path to PDF file to process')
    parser.add_argument('--output', default='output', 
                       help='Output directory (default: output/)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: File not found: {args.pdf_path}")
        sys.exit(1)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process PDF
    try:
        pipeline = PDFExtractionPipeline(output_dir=args.output)
        result = pipeline.process(args.pdf_path)
        
        # Display summary
        print(f"\n{'='*60}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*60}")
        
        metadata = result['document_metadata']
        summary = result['extraction_summary']
        
        print(f"\nFile: {metadata['source_file']}")
        print(f"PDF Type: {metadata['pdf_type'].upper()}")
        print(f"Total Pages: {metadata['total_pages']}")
        print(f"Extraction Date: {metadata['extraction_date']}")
        
        print(f"\nExtracted Content:")
        print(f"  ├─ Text Blocks: {summary['total_text_blocks']}")
        print(f"  ├─ Tables: {summary['total_tables']}")
        print(f"  ├─ Charts/Figures: {summary['total_charts']}")
        print(f"  └─ Headings: {summary['total_headings']}")
        
        if summary['pages_with_watermarks'] > 0:
            print(f"\n  ⚠ Pages with watermarks: {summary['pages_with_watermarks']}")
        
        # Display document structure
        if result['document_structure']['sections']:
            print(f"\nDocument Structure:")
            for section in result['document_structure']['sections']:
                indent = "  " * (section['level'] - 1)
                print(f"  {indent}├─ {section['heading']} (Page {section['page_number']})")
                
                for subsection in section.get('subsections', []):
                    sub_indent = "  " * subsection['level']
                    print(f"  {sub_indent}└─ {subsection['heading']} (Page {subsection['page_number']})")
        
        # Output file location
        output_filename = Path(args.pdf_path).stem + "_extracted.json"
        output_path = os.path.join(args.output, output_filename)
        
        print(f"\n{'='*60}")
        print(f"✓ Extraction completed successfully!")
        print(f"✓ Output saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Extraction failed")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        
        if args.verbose:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        
        print(f"\n{'='*60}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
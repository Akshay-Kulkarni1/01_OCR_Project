#!/usr/bin/env python3
"""
Mechanical Drawing PDF Data Extractor

Extracts structured data from mechanical construction drawing PDFs including:
- Tables (schedules, calculations, etc.)
- Text sections (notes, specifications)
- Tagged annotations (equipment tags, duct sizes)

Usage:
    python 
      input.pdf [output_directory]
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Install with: pip install PyMuPDF")
    exit(1)

try:
    import tabula
except ImportError:
    print("tabula-py not found. Install with: pip install tabula-py")
    tabula = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExtractedTable:
    """Represents an extracted table with metadata"""
    data: pd.DataFrame
    label: str
    page_number: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1

@dataclass 
class TextSection:
    """Represents a text section with heading"""
    heading: str
    content: List[str]
    page_number: int
    bbox: Tuple[float, float, float, float]

@dataclass
class TaggedAnnotation:
    """Represents a tagged annotation"""
    tag: str
    page_number: int
    bbox: Tuple[float, float, float, float]
    context: str

class MechanicalPDFExtractor:
    """Main class for extracting data from mechanical drawing PDFs"""
    
    def __init__(self, pdf_path: str, output_dir: str = None):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir) if output_dir else self.pdf_path.parent / f"{self.pdf_path.stem}_extracted"
        self.output_dir.mkdir(exist_ok=True)
        
        # Open PDF document
        self.doc = fitz.open(str(self.pdf_path))
        
        # Pattern definitions for various mechanical elements
        self.table_keywords = [
            r'schedule', r'calculation', r'ventilation', r'equipment', r'terminal',
            r'unit', r'hvac', r'mechanical', r'duct', r'air', r'cfm', r'capacity',
            r'pressure', r'flow', r'fan', r'motor', r'cooling', r'heating'
        ]
        
        self.note_headings = [
            r'general\s+notes?', r'mechanical\s+notes?', r'keyed?\s+notes?',
            r'specifications?', r'remarks?', r'instructions?', r'legend',
            r'abbreviations?', r'symbols?', r'code\s+requirements?'
        ]
        
        # Regex patterns for tagged annotations
        self.tag_patterns = [
            r'\b[A-Z]{2,4}-\d+(?:-\d+)*\b',  # Equipment tags: AHU-1, VAV-3-1
            r'\([A-Z]\)\s*\d+[Xx]\d+',       # Duct sizes: (R) 10X6
            r'\b\d+[Xx]\d+\b',               # Simple duct sizes: 12X8
            r'\bCFM\s*\d+',                  # CFM values
            r'\b\d+\s*CFM\b',                # CFM values (alt format)
            r'\bFPM\s*\d+',                  # FPM values
            r'\b\d+°F\b',                    # Temperature
            r'\b\d+\s*PSI?\b',               # Pressure
        ]
        
        # Results storage
        self.extracted_tables: List[ExtractedTable] = []
        self.text_sections: List[TextSection] = []
        self.tagged_annotations: List[TaggedAnnotation] = []

    def extract_all_data(self) -> Dict:
        """Extract all data from the PDF"""
        logger.info(f"Processing PDF: {self.pdf_path}")
        
        # Extract tables
        self._extract_tables()
        
        # Extract text sections
        self._extract_text_sections()
        
        # Extract tagged annotations
        self._extract_tagged_annotations()
        
        # Save results
        self._save_results()
        
        return self._get_summary()

    def _extract_tables(self):
        """Extract tables from all pages"""
        logger.info("Extracting tables...")
        
        # Method 1: Use tabula-py if available
        if tabula:
            self._extract_tables_tabula()
        
        # Method 2: Use PyMuPDF table detection as fallback
        self._extract_tables_pymupdf()
        
        logger.info(f"Found {len(self.extracted_tables)} tables")

    def _extract_tables_tabula(self):
        """Extract tables using tabula-py"""
        try:
            # Read all tables from PDF
            tables = tabula.read_pdf(str(self.pdf_path), pages='all', multiple_tables=True)
            
            for i, df in enumerate(tables):
                if df.empty or len(df) < 2:
                    continue
                    
                # Infer table label
                label = self._infer_table_label(df, page_number=1)
                
                table = ExtractedTable(
                    data=df,
                    label=label,
                    page_number=1,  # tabula doesn't provide page info easily
                    confidence=0.8,
                    bbox=(0, 0, 0, 0)  # bbox not available from tabula
                )
                
                self.extracted_tables.append(table)
                
        except Exception as e:
            logger.warning(f"Tabula extraction failed: {e}")

    def _extract_tables_pymupdf(self):
        """Extract tables using PyMuPDF"""
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            
            # Find tables using PyMuPDF
            tables = page.find_tables()
            
            for table in tables:
                try:
                    # Extract table data
                    table_data = table.extract()
                    
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    
                    # Clean the DataFrame
                    df = self._clean_dataframe(df)
                    
                    if df.empty:
                        continue
                    
                    # Infer table label
                    label = self._infer_table_label(df, page_num + 1, page)
                    
                    table_obj = ExtractedTable(
                        data=df,
                        label=label,
                        page_number=page_num + 1,
                        confidence=0.7,
                        bbox=table.bbox
                    )
                    
                    self.extracted_tables.append(table_obj)
                    
                except Exception as e:
                    logger.warning(f"Error extracting table on page {page_num + 1}: {e}")

    def _extract_text_sections(self):
        """Extract text sections with headings"""
        logger.info("Extracting text sections...")
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            
            # Find text sections
            sections = self._find_text_sections(text, page_num + 1)
            self.text_sections.extend(sections)
        
        logger.info(f"Found {len(self.text_sections)} text sections")

    def _extract_tagged_annotations(self):
        """Extract tagged annotations from all pages"""
        logger.info("Extracting tagged annotations...")
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            
            # Get text with position information
            text_dict = page.get_text("dict")
            
            # Extract annotations
            annotations = self._find_tagged_annotations(text_dict, page_num + 1)
            self.tagged_annotations.extend(annotations)
        
        logger.info(f"Found {len(self.tagged_annotations)} tagged annotations")

    def _infer_table_label(self, df: pd.DataFrame, page_number: int, page=None) -> str:
        """Infer table label based on content and context"""
        # Check column headers
        headers = ' '.join(str(col).lower() for col in df.columns if pd.notna(col))
        
        # Check first few rows
        content = ' '.join(str(cell).lower() for row in df.head(3).values 
                          for cell in row if pd.notna(cell))
        
        combined_text = headers + ' ' + content
        
        # Match against common mechanical table types
        if any(keyword in combined_text for keyword in ['schedule', 'equipment']):
            return "Equipment_Schedule"
        elif any(keyword in combined_text for keyword in ['ventilation', 'cfm', 'air']):
            return "Ventilation_Schedule"
        elif any(keyword in combined_text for keyword in ['terminal', 'vav', 'cav']):
            return "Terminal_Unit_Schedule"
        elif any(keyword in combined_text for keyword in ['calculation', 'load']):
            return "Load_Calculations"
        elif any(keyword in combined_text for keyword in ['duct', 'size']):
            return "Duct_Schedule"
        elif any(keyword in combined_text for keyword in ['motor', 'fan']):
            return "Motor_Fan_Schedule"
        else:
            return f"Table_Page_{page_number}"

    def _find_text_sections(self, text: str, page_number: int) -> List[TextSection]:
        """Find and extract text sections with headings"""
        sections = []
        lines = text.split('\n')
        
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a heading
            is_heading = False
            heading_match = None
            
            for pattern in self.note_headings:
                if re.search(pattern, line, re.IGNORECASE):
                    is_heading = True
                    heading_match = line
                    break
            
            # Also check for numbered headings or all caps headings
            if not is_heading:
                if (re.match(r'^\d+\.', line) or 
                    (line.isupper() and len(line.split()) <= 5 and len(line) > 5)):
                    is_heading = True
                    heading_match = line
            
            if is_heading:
                # Save previous section
                if current_section and current_content:
                    sections.append(TextSection(
                        heading=current_section,
                        content=current_content,
                        page_number=page_number,
                        bbox=(0, 0, 0, 0)  # Approximate
                    ))
                
                # Start new section
                current_section = heading_match
                current_content = []
            else:
                # Add to current section
                if current_section:
                    current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            sections.append(TextSection(
                heading=current_section,
                content=current_content,
                page_number=page_number,
                bbox=(0, 0, 0, 0)
            ))
        
        return sections

    def _find_tagged_annotations(self, text_dict: dict, page_number: int) -> List[TaggedAnnotation]:
        """Find tagged annotations in text"""
        annotations = []
        
        # Traverse text blocks
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"]
                    bbox = span["bbox"]
                    
                    # Check each pattern
                    for pattern in self.tag_patterns:
                        matches = re.finditer(pattern, text, re.IGNORECASE)
                        for match in matches:
                            tag = match.group()
                            
                            # Get context (surrounding text)
                            start = max(0, match.start() - 20)
                            end = min(len(text), match.end() + 20)
                            context = text[start:end].strip()
                            
                            annotation = TaggedAnnotation(
                                tag=tag,
                                page_number=page_number,
                                bbox=bbox,
                                context=context
                            )
                            
                            annotations.append(annotation)
        
        return annotations

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize DataFrame"""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = [str(col).strip() if pd.notna(col) else f"Column_{i}" 
                     for i, col in enumerate(df.columns)]
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df

    def _save_results(self):
        """Save all extracted data to files"""
        logger.info(f"Saving results to {self.output_dir}")
        
        # Save tables
        self._save_tables()
        
        # Save text sections
        self._save_text_sections()
        
        # Save tagged annotations
        self._save_tagged_annotations()
        
        # Save summary
        self._save_summary()

    def _save_tables(self):
        """Save extracted tables to CSV files"""
        tables_dir = self.output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        
        table_metadata = []
        
        for i, table in enumerate(self.extracted_tables):
            # Save CSV
            filename = f"{table.label}_{i+1}.csv"
            csv_path = tables_dir / filename
            table.data.to_csv(csv_path, index=False)
            
            # Save metadata
            metadata = {
                "filename": filename,
                "label": table.label,
                "page_number": table.page_number,
                "confidence": table.confidence,
                "rows": len(table.data),
                "columns": len(table.data.columns),
                "column_names": list(table.data.columns)
            }
            table_metadata.append(metadata)
        
        # Save table metadata
        with open(tables_dir / "tables_metadata.json", 'w') as f:
            json.dump(table_metadata, f, indent=2)

    def _save_text_sections(self):
        """Save text sections to JSON"""
        sections_data = {}
        
        for section in self.text_sections:
            # Group by heading
            if section.heading not in sections_data:
                sections_data[section.heading] = []
            
            sections_data[section.heading].append({
                "content": section.content,
                "page_number": section.page_number
            })
        
        # Save to JSON
        with open(self.output_dir / "text_sections.json", 'w') as f:
            json.dump(sections_data, f, indent=2)

    def _save_tagged_annotations(self):
        """Save tagged annotations to JSON and CSV"""
        # Convert to list of dictionaries
        annotations_data = [asdict(ann) for ann in self.tagged_annotations]
        
        # Save as JSON
        with open(self.output_dir / "tagged_annotations.json", 'w') as f:
            json.dump(annotations_data, f, indent=2)
        
        # Save as CSV for easy analysis
        if annotations_data:
            df = pd.DataFrame(annotations_data)
            df = df.drop('bbox', axis=1)  # Remove bbox for CSV simplicity
            df.to_csv(self.output_dir / "tagged_annotations.csv", index=False)

    def _save_summary(self):
        """Save extraction summary"""
        summary = self._get_summary()
        
        with open(self.output_dir / "extraction_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

    def _get_summary(self) -> Dict:
        """Get summary of extraction results"""
        return {
            "pdf_file": str(self.pdf_path),
            "total_pages": len(self.doc),
            "extraction_summary": {
                "tables_extracted": len(self.extracted_tables),
                "text_sections_extracted": len(self.text_sections),
                "tagged_annotations_extracted": len(self.tagged_annotations)
            },
            "table_labels": [table.label for table in self.extracted_tables],
            "text_section_headings": list(set(section.heading for section in self.text_sections)),
            "unique_tag_types": list(set(ann.tag for ann in self.tagged_annotations)),
            "output_directory": str(self.output_dir)
        }

    def close(self):
        """Close the PDF document"""
        if hasattr(self, 'doc'):
            self.doc.close()

def main():
    parser = argparse.ArgumentParser(description="Extract structured data from mechanical drawing PDFs")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: PDF_filename_extracted)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF file not found: {args.pdf_path}")
        return 1
    
    try:
        # Create extractor and process PDF
        extractor = MechanicalPDFExtractor(args.pdf_path, args.output_dir)
        summary = extractor.extract_all_data()
        
        # Print summary
        print("\n" + "="*50)
        print("EXTRACTION SUMMARY")
        print("="*50)
        print(f"PDF File: {summary['pdf_file']}")
        print(f"Total Pages: {summary['total_pages']}")
        print(f"Tables Extracted: {summary['extraction_summary']['tables_extracted']}")
        print(f"Text Sections Extracted: {summary['extraction_summary']['text_sections_extracted']}")
        print(f"Tagged Annotations Extracted: {summary['extraction_summary']['tagged_annotations_extracted']}")
        print(f"Output Directory: {summary['output_directory']}")
        
        if summary['table_labels']:
            print(f"\nTable Types Found:")
            for label in summary['table_labels']:
                print(f"  - {label}")
        
        if summary['text_section_headings']:
            print(f"\nText Section Headings:")
            for heading in summary['text_section_headings']:
                print(f"  - {heading}")
        
        print("\nExtraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return 1
    
    finally:
        if 'extractor' in locals():
            extractor.close()
    
    return 0

if __name__ == "__main__":
    exit(main())
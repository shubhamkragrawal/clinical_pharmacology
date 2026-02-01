"""
Section Analyzer & Visualizer
Helps you understand and work with parsed PDF sections
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import re


class SectionAnalyzer:
    """Analyze and visualize PDF sections"""
    
    def __init__(self, json_path: str):
        """Load parsed PDF JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.filename = self.data.get('filename', '')
        self.sections = self.data.get('sections', [])
        self.full_text = self.data.get('full_text_content', '')
    
    def print_section_outline(self):
        """Print a hierarchical outline of all sections"""
        print(f"\n{'='*80}")
        print(f"DOCUMENT: {self.filename}")
        print(f"{'='*80}")
        print(f"Total Sections: {len(self.sections)}")
        print(f"Total Text Length: {len(self.full_text):,} characters")
        print(f"{'='*80}\n")
        
        for i, section in enumerate(self.sections, 1):
            level = section.get('section_level', 1)
            title = section.get('section_title', 'Untitled')
            content_length = section.get('content_length', 0)
            page = section.get('page_number', -1)
            
            # Indent based on level
            indent = "  " * (level - 1)
            
            # Format output
            page_info = f"(Page {page})" if page > 0 else ""
            print(f"{i}. {indent}{'▸ ' * level}{title} {page_info}")
            print(f"   {indent}├─ Level: {level}")
            print(f"   {indent}├─ Content Length: {content_length:,} chars")
            print(f"   {indent}└─ Preview: {self._get_preview(section.get('section_content', ''), 100)}")
            print()
    
    def _get_preview(self, text: str, max_length: int = 100) -> str:
        """Get preview of text"""
        text = text.strip()
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def get_section_by_title(self, title_keyword: str) -> List[Dict]:
        """Find sections containing a keyword in the title"""
        matching = []
        for section in self.sections:
            if title_keyword.lower() in section.get('section_title', '').lower():
                matching.append(section)
        return matching
    
    def get_section_by_level(self, level: int) -> List[Dict]:
        """Get all sections at a specific level"""
        return [s for s in self.sections if s.get('section_level') == level]
    
    def export_sections_to_csv(self, output_path: str):
        """Export sections to CSV for easy viewing"""
        records = []
        for i, section in enumerate(self.sections, 1):
            records.append({
                'section_number': i,
                'title': section.get('section_title', ''),
                'level': section.get('section_level', 1),
                'content_preview': section.get('section_content', '')[:200],
                'content_length': section.get('content_length', 0),
                'page_number': section.get('page_number', -1),
                'start_position': section.get('start_position', -1)
            })
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"Sections exported to: {output_path}")
    
    def export_section_content(self, section_index: int, output_path: str):
        """Export a specific section's content to a text file"""
        if 0 <= section_index < len(self.sections):
            section = self.sections[section_index]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"SECTION: {section.get('section_title', 'Untitled')}\n")
                f.write(f"Level: {section.get('section_level', 1)}\n")
                f.write(f"Page: {section.get('page_number', -1)}\n")
                f.write(f"{'='*80}\n\n")
                f.write(section.get('section_content', ''))
            
            print(f"Section content exported to: {output_path}")
        else:
            print(f"Error: Section index {section_index} out of range (0-{len(self.sections)-1})")
    
    def create_markdown_outline(self, output_path: str):
        """Create a Markdown file with the document outline"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.filename}\n\n")
            f.write(f"**Total Sections:** {len(self.sections)}  \n")
            f.write(f"**Total Length:** {len(self.full_text):,} characters  \n\n")
            f.write("---\n\n")
            
            for i, section in enumerate(self.sections, 1):
                level = section.get('section_level', 1)
                title = section.get('section_title', 'Untitled')
                content = section.get('section_content', '')
                
                # Use markdown heading levels
                heading = "#" * min(level + 1, 6)  # Max 6 levels in markdown
                
                f.write(f"{heading} {title}\n\n")
                f.write(f"{content}\n\n")
                f.write("---\n\n")
        
        print(f"Markdown outline created: {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the document sections"""
        if not self.sections:
            return {
                'total_sections': 0,
                'avg_content_length': 0,
                'max_level': 0,
                'sections_per_level': {}
            }
        
        content_lengths = [s.get('content_length', 0) for s in self.sections]
        levels = [s.get('section_level', 1) for s in self.sections]
        
        from collections import Counter
        level_counts = Counter(levels)
        
        return {
            'total_sections': len(self.sections),
            'avg_content_length': sum(content_lengths) / len(content_lengths),
            'min_content_length': min(content_lengths),
            'max_content_length': max(content_lengths),
            'max_level': max(levels),
            'sections_per_level': dict(level_counts)
        }


def analyze_batch_pdfs(json_directory: str, output_summary_csv: str):
    """Analyze all parsed PDFs in a directory and create summary"""
    json_files = list(Path(json_directory).glob('*_parsed.json'))
    
    summary_records = []
    
    for json_file in json_files:
        try:
            analyzer = SectionAnalyzer(str(json_file))
            stats = analyzer.get_statistics()
            
            summary_records.append({
                'filename': analyzer.filename,
                'json_path': str(json_file),
                'total_sections': stats['total_sections'],
                'max_section_level': stats['max_level'],
                'avg_content_length': round(stats['avg_content_length'], 2),
                'min_content_length': stats['min_content_length'],
                'max_content_length': stats['max_content_length'],
                'level_1_sections': stats['sections_per_level'].get(1, 0),
                'level_2_sections': stats['sections_per_level'].get(2, 0),
                'level_3_sections': stats['sections_per_level'].get(3, 0),
            })
        except Exception as e:
            print(f"Error analyzing {json_file}: {e}")
    
    df = pd.DataFrame(summary_records)
    df.to_csv(output_summary_csv, index=False)
    print(f"\nBatch analysis complete!")
    print(f"Analyzed {len(summary_records)} files")
    print(f"Summary saved to: {output_summary_csv}")
    
    return df


def search_content_across_pdfs(json_directory: str, search_term: str):
    """Search for a term across all parsed PDFs"""
    json_files = list(Path(json_directory).glob('*_parsed.json'))
    
    results = []
    
    for json_file in json_files:
        try:
            analyzer = SectionAnalyzer(str(json_file))
            
            # Search in sections
            for i, section in enumerate(analyzer.sections):
                title = section.get('section_title', '')
                content = section.get('section_content', '')
                
                if (search_term.lower() in title.lower() or 
                    search_term.lower() in content.lower()):
                    
                    results.append({
                        'filename': analyzer.filename,
                        'section_number': i + 1,
                        'section_title': title,
                        'section_level': section.get('section_level', 1),
                        'match_in': 'title' if search_term.lower() in title.lower() else 'content'
                    })
        except Exception as e:
            print(f"Error searching {json_file}: {e}")
    
    if results:
        df = pd.DataFrame(results)
        print(f"\nFound {len(results)} matches for '{search_term}':")
        print(df.to_string(index=False))
        return df
    else:
        print(f"No matches found for '{search_term}'")
        return None


# Example usage and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze parsed PDF sections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Analyze a single PDF:
   python section_analyzer.py single document_parsed.json

2. Export section outline to CSV:
   python section_analyzer.py export-csv document_parsed.json sections.csv

3. Create markdown outline:
   python section_analyzer.py markdown document_parsed.json outline.md

4. Analyze all PDFs in directory:
   python section_analyzer.py batch ./output_jsons summary.csv

5. Search across all PDFs:
   python section_analyzer.py search ./output_jsons "risk assessment"

6. Export specific section:
   python section_analyzer.py export-section document_parsed.json 0 section_0.txt
        """
    )
    
    parser.add_argument('command', choices=[
        'single', 'export-csv', 'markdown', 'batch', 'search', 'export-section'
    ])
    parser.add_argument('input', help='JSON file or directory')
    parser.add_argument('output', nargs='?', help='Output file or search term')
    parser.add_argument('section_index', nargs='?', type=int, help='Section index for export-section')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        analyzer = SectionAnalyzer(args.input)
        analyzer.print_section_outline()
        stats = analyzer.get_statistics()
        print("\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.command == 'export-csv':
        analyzer = SectionAnalyzer(args.input)
        analyzer.export_sections_to_csv(args.output)
    
    elif args.command == 'markdown':
        analyzer = SectionAnalyzer(args.input)
        analyzer.create_markdown_outline(args.output)
    
    elif args.command == 'batch':
        analyze_batch_pdfs(args.input, args.output)
    
    elif args.command == 'search':
        search_content_across_pdfs(args.input, args.output)
    
    elif args.command == 'export-section':
        if args.section_index is None:
            print("Error: section_index required for export-section command")
        else:
            analyzer = SectionAnalyzer(args.input)
            analyzer.export_section_content(args.section_index, args.output)
"""
Parallel PDF Processor
Process multiple PDFs using multiple CPU cores
"""

import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import logging
from datetime import datetime

# Setup logging for parallel processor
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/parallel_process_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_single_pdf(args):
    """Process a single PDF - used by parallel processor"""
    pdf_path, output_dir = args
    
    try:
        # Import here to avoid issues with multiprocessing
        from pdf_extraction_pipeline import PDFExtractionPipeline
        
        logger.info(f"Starting: {pdf_path.name}")
        pipeline = PDFExtractionPipeline(output_dir=output_dir)
        result = pipeline.process(str(pdf_path))
        
        # Verify output file exists
        output_file = os.path.join(output_dir, f"{pdf_path.stem}_extracted.json")
        if not os.path.exists(output_file):
            raise IOError(f"Output file not created: {output_file}")
        
        file_size = os.path.getsize(output_file)
        logger.info(f"Completed: {pdf_path.name} ({file_size:,} bytes)")
        
        return (True, pdf_path.name, None)
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Failed: {pdf_path.name} - {error_msg}")
        return (False, pdf_path.name, error_msg)


def parallel_process(input_dir, output_dir, num_workers=None):
    """Process PDFs in parallel"""
    
    # Find all PDFs
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        logger.warning(f"No PDF files found in {input_dir}")
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    logger.info(f"Using {num_workers} parallel workers")
    logger.info(f"Output directory: {output_dir}")
    
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Using {num_workers} parallel workers")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tasks
    tasks = [(pdf, output_dir) for pdf in pdf_files]
    
    # Process in parallel
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_pdf, tasks),
            total=len(tasks),
            desc="Processing PDFs"
        ))
    
    # Calculate statistics
    successful = sum(1 for r in results if r[0])
    failed = sum(1 for r in results if not r[0])
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "="*60)
    print("PARALLEL PROCESSING COMPLETE")
    print("="*60)
    print(f"Total: {len(pdf_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Speed: {len(pdf_files)/(elapsed/60):.1f} PDFs/minute")
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    
    # Show failures
    if failed > 0:
        print("\nFailed files:")
        logger.error("Failed files:")
        for success, name, error in results:
            if not success:
                print(f"  - {name}: {error}")
                logger.error(f"  - {name}: {error}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "processing_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Processing Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total files: {len(pdf_files)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Time: {elapsed/60:.1f} minutes\n")
        f.write(f"\nFailed files:\n")
        for success, name, error in results:
            if not success:
                f.write(f"  - {name}: {error}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parallel PDF Processing')
    parser.add_argument('input_dir', help='Input directory')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    logger.info(f"Starting parallel processing")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output}")
    
    parallel_process(args.input_dir, args.output, args.workers)

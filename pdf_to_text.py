import os
from PyPDF2 import PdfReader
import json

def convert_pdf_to_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error converting {pdf_path}: {str(e)}")
        return None

def process_pdf_directory(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each PDF file
    results = {}
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            text = convert_pdf_to_text(pdf_path)
            
            if text:
                # Save as text file
                txt_filename = filename.replace('.pdf', '.txt')
                txt_path = os.path.join(output_dir, txt_filename)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # Store in results dictionary
                results[filename] = {
                    'status': 'success',
                    'text_path': txt_path,
                    'line_count': len(text.split('\n'))
                }
            else:
                results[filename] = {
                    'status': 'failed',
                    'error': 'Conversion failed'
                }
    
    # Save results summary
    with open(os.path.join(output_dir, 'conversion_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    input_dir = "trainingdata/pdf"
    output_dir = "trainingdata/text"
    results = process_pdf_directory(input_dir, output_dir)
    print("PDF conversion completed. Results saved in:", output_dir) 
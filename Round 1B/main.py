import json
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModel
import torch
import spacy
from collections import Counter
import re
import numpy as np

# --- Dependencies ---
# pip install PyMuPDF transformers torch spacy
# python -m spacy download en_core_web_sm

class PDFAnalyzer:
    """
    Analyzes a PDF document to extract its structural sections based on layout cues.
    
    This class implements a multi-pass strategy:
    1. Extract all text blocks with their font metadata.
    2. Determine a baseline paragraph style (size, font).
    3. Classify blocks as headings or paragraphs based on deviations from the baseline.
    4. Group paragraphs under their corresponding headings to form logical sections.
    """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.min_heading_size = 11  # Minimum font size to be considered a heading

    def _get_baseline_style(self, blocks: List[Dict[str, Any]]) -> Tuple[float, str]:
        """Calculates the most common font size and name for the body text."""
        if not blocks:
            return 10.0, "Helvetica" # Return a sensible default

        sizes = [span['size'] for block in blocks for line in block['lines'] for span in line['spans']]
        fonts = [span['font'] for block in blocks for line in block['lines'] for span in line['spans']]
        
        # Most common font size is likely the paragraph text
        baseline_size = Counter(sizes).most_common(1)[0][0] if sizes else 10.0
        baseline_font = Counter(fonts).most_common(1)[0][0] if fonts else "Helvetica"
        
        return baseline_size, baseline_font

    def _classify_block(self, block: Dict[str, Any], baseline_size: float) -> str:
        """Classifies a text block as 'heading' or 'paragraph'."""
        if not block['lines']:
            return 'paragraph' # Or skip if preferred

        block_text = " ".join(
            span['text'].strip() for line in block['lines'] for span in line['spans']
        ).strip()

        if not block_text:
            return 'paragraph' # Treat empty blocks as paragraphs

        # Heuristic 1: Font Size
        avg_size = np.mean([span['size'] for line in block['lines'] for span in line['spans']])
        if avg_size < max(self.min_heading_size, baseline_size * 1.15):
            return 'paragraph'

        # Heuristic 2: Text Length (headings are typically short)
        if len(block_text.split()) > 15:
            return 'paragraph'

        # Heuristic 3: Content (headings rarely contain full sentences with verbs)
        # Also check for list-like patterns which are usually not headings
        if re.match(r'^\s*[\*\-•\d+]\.?\s+', block_text):
            return 'paragraph'

        doc = self.nlp(block_text)
        if any(token.pos_ == "VERB" for token in doc):
            return 'paragraph'
            
        return 'heading'

    def extract_structured_sections(self, doc_path: str) -> List[Dict[str, Any]]:
        """
        Extracts logically grouped sections from a PDF document.
        Returns: List of sections with their titles, content, and page numbers.
        """
        doc = fitz.open(doc_path)
        sections = []
        current_section = None
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                        
                    block_text = " ".join(
                        span['text'].strip() 
                        for line in block['lines'] 
                        for span in line['spans']
                    ).strip()
                    
                    if not block_text:
                        continue
                    
                    is_heading = self._is_potential_heading(block)
                    
                    if is_heading:
                        if current_section and current_section['text']:
                            sections.append(current_section)
                        current_section = {
                            "title": block_text,
                            "text": "",
                            "page": page_num + 1
                        }
                    elif current_section:
                        current_section['text'] += f" {block_text}"
                    else:
                        # Start a new section if we encounter text before any heading
                        current_section = {
                            "title": "Introduction",
                            "text": block_text,
                            "page": page_num + 1
                        }
            
            # Add the last section
            if current_section and current_section['text']:
                sections.append(current_section)
                
        finally:
            doc.close()
            
        return sections
    
    def _is_potential_heading(self, block: Dict) -> bool:
        """
        Determines if a block is likely a heading based on font properties.
        """
        if not block.get('lines'):
            return False
            
        spans = [span for line in block['lines'] for span in line['spans']]
        if not spans:
            return False
            
        text = " ".join(span['text'].strip() for span in spans)
        
        # Heading heuristics
        is_bold = any('bold' in span.get('font', '').lower() for span in spans)
        avg_size = sum(span['size'] for span in spans) / len(spans)
        is_large = avg_size >= 11  # Minimum heading size
        is_short = len(text.split()) <= 12  # Headings are typically short
        
        return (is_bold or is_large) and is_short

class DocumentAnalyzer:
    """
    Processes a collection of documents to extract and rank sections
    based on a persona and a job-to-be-done.
    """
    def _init_(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # This model is small (~86MB), fast, and great for semantic similarity.
        # It perfectly fits the constraints.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.pdf_analyzer = PDFAnalyzer()
        
        # Add semantic importance thresholds
        self.importance_thresholds = {
            'critical': 0.7,
            'high': 0.5,
            'medium': 0.3,
            'low': 0.1
        }

    def _calculate_importance(self, section: Dict, persona: str, job_description: str) -> float:
        """Enhanced importance calculation using multiple factors"""
        # Create contextual query
        query = f"As a {persona}, {job_description}"
        
        # Calculate multiple relevance scores
        title_score = self._calculate_semantic_similarity(section['title'], query)
        content_score = self._calculate_semantic_similarity(section['text'], query)
        
        # Additional relevance factors
        keyword_score = self._calculate_keyword_relevance(section, persona, job_description)
        position_score = self._calculate_position_importance(section.get('page', 1))
        
        # Weighted combination of scores
        final_score = (
            0.4 * title_score +      # Title relevance
            0.3 * content_score +    # Content relevance
            0.2 * keyword_score +    # Keyword matching
            0.1 * position_score     # Position importance
        )
        
        return round(float(final_score), 4)

    def _calculate_semantic_similarity(self, text: str, query: str) -> float:
        """Calculate semantic similarity using transformer embeddings"""
        inputs = self.tokenizer(
            [text, query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
        return float(similarity.item())

    def _calculate_keyword_relevance(self, section: Dict, persona: str, job: str) -> float:
        """Calculate relevance based on keyword matching"""
        # Extract key terms from persona and job
        key_terms = set(self._extract_key_terms(persona + " " + job))
        
        # Extract terms from section
        section_terms = set(self._extract_key_terms(section['title'] + " " + section['text']))
        
        # Calculate Jaccard similarity
        if not key_terms or not section_terms:
            return 0.0
            
        intersection = len(key_terms.intersection(section_terms))
        union = len(key_terms.union(section_terms))
        
        return intersection / union if union > 0 else 0.0

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract important terms using spacy"""
        doc = self.pdf_analyzer.nlp(text.lower())
        return [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'VERB', 'ADJ']
        ]

    def _calculate_position_importance(self, page_num: int, max_pages: int = 10) -> float:
        """Calculate importance based on position in document"""
        # Earlier pages often contain more important information
        return max(1.0 - (page_num / max_pages), 0.1)

    def _refine_text(self, text: str) -> str:
        """Performs basic text cleaning."""
        # Consolidate whitespace
        text = " ".join(text.split())
        # Further refinement could handle hyphenation, special characters, etc.
        return text

    def process_documents(self, doc_paths: List[str], persona: str, job_description: str) -> Dict:
        """
        Main processing pipeline for a collection of documents.
        """
        results = {
            "metadata": {
                "input_documents": [os.path.basename(p) for p in doc_paths],
                "persona": persona,
                "job_to_be_done": job_description,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        
        # Process each document
        for doc_path in doc_paths:
            doc_name = os.path.basename(doc_path)
            print(f"Processing {doc_name}...")
            
            sections = self.pdf_analyzer.extract_structured_sections(doc_path)
            
            # Pre-calculate document-level features
            doc_sections_count = len(sections)
            
            for idx, section in enumerate(sections):
                # Calculate importance with enhanced method
                importance = self._calculate_importance(section, persona, job_description)
                
                # Filter out low-relevance sections
                if importance >= self.importance_thresholds['low']:
                    results["extracted_sections"].append({
                        "document": doc_name,
                        "page_number": section["page"],
                        "section_title": section["title"],
                        "importance_rank": importance
                    })
                    
                    results["subsection_analysis"].append({
                        "document": doc_name,
                        "page_number": section["page"],
                        "section_title": section["title"],
                        "refined_text": self._refine_text(section["text"])
                    })
        
        # Sort sections by importance rank
        results["extracted_sections"].sort(
            key=lambda x: x["importance_rank"], 
            reverse=True
        )
        
        # Keep only top N most relevant sections
        max_sections = 20  # Adjust based on needs
        results["extracted_sections"] = results["extracted_sections"][:max_sections]
        
        return results

# --- Main Execution Logic (remains largely the same) ---
def load_test_case(collection_path: str) -> Dict:
    """Loads test case from the input JSON file."""
    # Move up one level from PDFs folder to get to collection root
    collection_root = os.path.dirname(collection_path)
    input_file = os.path.join(collection_root, "challenge1b_input.json")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    """Main function to run the analysis on specified collections."""
    analyzer = DocumentAnalyzer()
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    for i in range(1, 4):
        # Update path to include PDFs subfolder
        collection_path = os.path.join(base_path, f"Collection {i}", "PDFs")
        print(f"\nProcessing Collection {i}")
        print(f"Collection path: {collection_path}")
        
        if not os.path.exists(collection_path):
            print(f"Error: PDFs folder not found at {collection_path}")
            continue

        try:
            test_case = load_test_case(collection_path)
            
            challenge_info = test_case["challenge_info"]
            persona = test_case["persona"]["role"]
            job = test_case["job_to_be_done"]["task"]
            
            print(f"Found {len(test_case['documents'])} documents in input JSON")
            
            # Update PDF paths to use PDFs subfolder
            pdf_files = []
            for doc in test_case["documents"]:
                pdf_path = os.path.join(collection_path, doc["filename"])
                if os.path.exists(pdf_path):
                    pdf_files.append(pdf_path)
                    print(f"Found PDF: {doc['filename']}")
                else:
                    print(f"Missing PDF: {doc['filename']}")
            
            if not pdf_files:
                print(f"Error: No PDF documents found in {collection_path}")
                print("Expected structure:")
                print(f"{collection_path}/")
                print("├── challenge1b_input.json")
                print("└── [PDF files listed in input.json]")
                continue
            
            print(f"\nProcessing {len(pdf_files)} PDF files...")    
            results = analyzer.process_documents(pdf_files, persona, job)
            
            # Augment metadata with challenge info
            results["metadata"].update(challenge_info)
            
            output_file = os.path.join(
                base_path,
                f"output_collection_{i}_{challenge_info['challenge_id']}.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"Success: Output saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing Collection {i}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
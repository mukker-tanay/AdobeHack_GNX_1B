This project analyzes PDF documents to extract structured sections and rank them by importance based on a given **persona** and **job-to-be-done**. 
It uses font and layout cues for sectioning and applies semantic relevance scoring via transformer-based embeddings.

## Key Features
- Extracts structured sections (titles, text, page number) from PDFs
- Identifies and ranks relevant sections using semantic similarity and heuristics
- Supports batch processing for multiple document collections
- Uses spaCy, PyMuPDF, Hugging Face Transformers, and Torch
- Outputs JSON files with ranked sections and refined content

## Use Case

This tool is useful for challenges like summarizing legal, technical, or corporate documents to extract only the most relevant insights for a user role (e.g., compliance officer, hiring manager, analyst).

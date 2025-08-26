#!/usr/bin/env python3
"""
LLM Document Processor

A tool that processes documents using AI according to a given prompt.
Can be used standalone or imported as a module.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import PyPDF2
from transformers import pipeline, AutoProcessor, AutoModelForCausalLM
import torch


class LLMDocumentProcessor:
    def __init__(self):
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the Llama 3.2 Vision model"""
        try:
            model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback text generation pipeline")
            self.model = pipeline("text-generation", model="microsoft/DialoGPT-medium")
    
    def _read_text_file(self, filepath: str) -> str:
        """Read content from a text file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            raise ValueError(f"Error reading text file {filepath}: {e}")
    
    def _read_pdf_file(self, filepath: str) -> str:
        """Extract text content from a PDF file"""
        try:
            text = ""
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise ValueError(f"Error reading PDF file {filepath}: {e}")
    
    def _read_document(self, filepath: str) -> str:
        """Read content from a document (text or PDF)"""
        file_path = Path(filepath)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if file_path.suffix.lower() == '.pdf':
            return self._read_pdf_file(filepath)
        else:
            return self._read_text_file(filepath)
    
    def _generate_response(self, prompt: str, documents: List[str]) -> str:
        """Generate AI response using the prompt and documents"""
        combined_input = f"{prompt}\n\nDocuments to process:\n\n"
        
        for i, doc_content in enumerate(documents, 1):
            combined_input += f"Document {i}:\n{doc_content}\n\n"
        
        try:
            if hasattr(self.model, 'generate'):
                inputs = self.processor(combined_input, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=2000,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                return response[len(combined_input):].strip()
            else:
                result = self.model(combined_input, max_length=len(combined_input) + 500, num_return_sequences=1)
                return result[0]['generated_text'][len(combined_input):].strip()
        except Exception as e:
            return f"Error generating response: {e}\n\nInput was:\n{combined_input}"
    
    def _save_markdown(self, content: str) -> str:
        """Save the AI response to a markdown file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"llm_output_{timestamp}.md"
        output_path = Path(output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(content)
            return str(output_path.absolute())
        except Exception as e:
            raise ValueError(f"Error saving markdown file: {e}")
    
    def process_documents(self, file_paths: List[str]) -> str:
        """
        Process documents according to a prompt and return the output file path.
        
        Args:
            file_paths: List of file paths where the first is the prompt file
                       and the rest are documents to process
        
        Returns:
            Path to the generated markdown file
        """
        if not file_paths or len(file_paths) < 1:
            raise ValueError("At least one file path (prompt file) must be provided")
        
        prompt_file = file_paths[0]
        document_files = file_paths[1:]
        
        prompt = self._read_document(prompt_file)
        
        documents = []
        for doc_file in document_files:
            doc_content = self._read_document(doc_file)
            documents.append(doc_content)
        
        response = self._generate_response(prompt, documents)
        output_path = self._save_markdown(response)
        
        return output_path


def get_file_paths_interactive() -> List[str]:
    """Get file paths interactively from user input"""
    file_paths = []
    print("LLM Document Processor")
    print("Enter file paths one by one (press Enter with no input to finish):")
    
    while True:
        if not file_paths:
            filepath = input("Prompt file path: ").strip()
        else:
            filepath = input(f"Document file path {len(file_paths)}: ").strip()
        
        if not filepath:
            break
        
        file_paths.append(filepath)
    
    return file_paths


def process_documents(file_paths: List[str]) -> str:
    """
    Process documents using AI according to a prompt.
    
    Args:
        file_paths: List of file paths where the first is the prompt file
                   and the rest are documents to process
    
    Returns:
        Path to the generated markdown file
    """
    processor = LLMDocumentProcessor()
    return processor.process_documents(file_paths)


def main():
    """Main function for standalone execution"""
    try:
        file_paths = get_file_paths_interactive()
        
        if not file_paths:
            print("No files provided. Exiting.")
            return
        
        print(f"\nProcessing {len(file_paths)} files...")
        output_path = process_documents(file_paths)
        print(f"Output saved to: {output_path}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
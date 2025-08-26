#!/usr/bin/env python3
"""
LLM Document Processor

A tool that processes documents using AI according to a given prompt.
Can be used standalone or imported as a module.
"""

import os
import sys
import logging
import argparse
import glob
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import PyPDF2
import ollama
from tqdm import tqdm


class Logger:
    def __init__(self, verbose: bool = False, cleanup_logs: bool = False):
        self.verbose = verbose
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        if cleanup_logs:
            self._cleanup_logs()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"llm_processor_{timestamp}.log"
        
        self.logger = logging.getLogger('llm_processor')
        self.logger.setLevel(logging.DEBUG)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def _cleanup_logs(self):
        """Remove all existing log files"""
        log_files = glob.glob(str(self.log_dir / "*.log"))
        for log_file in log_files:
            try:
                os.remove(log_file)
            except Exception as e:
                pass
    
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def user_interaction(self, prompt_msg: str, response: str):
        """Log user interactions (prompt and response)"""
        self.logger.info(f"USER_PROMPT: {prompt_msg}")
        self.logger.info(f"USER_RESPONSE: {response}")
    
    def system_output(self, message: str):
        """Log system output that is shown to user"""
        self.logger.info(f"SYSTEM_OUTPUT: {message}")


class InteractiveIO:
    """Wrapper for interactive input/output with logging"""
    def __init__(self, logger: Logger):
        self.logger = logger
    
    def print(self, message: str):
        """Print message to console and log it"""
        print(message)
        self.logger.system_output(message)
    
    def input(self, prompt: str) -> str:
        """Get user input and log the interaction"""
        user_response = input(prompt).strip()
        self.logger.user_interaction(prompt.rstrip(), user_response)
        return user_response


class ProgressBar:
    def __init__(self, total_steps: int, verbose: bool = False):
        self.total_steps = total_steps
        self.current_step = 0
        self.verbose = verbose
        self.pbar = None
        
        if not verbose:
            self.pbar = tqdm(total=total_steps, desc="Processing", unit="step")
    
    def update(self, description: str = None):
        self.current_step += 1
        if not self.verbose and self.pbar:
            if description:
                self.pbar.set_description(description)
            self.pbar.update(1)
    
    def close(self):
        if not self.verbose and self.pbar:
            self.pbar.close()


class LLMDocumentProcessor:
    def __init__(self, logger: Logger = None, verbose: bool = False):
        self.model_name = 'llama3.2-vision'
        self.logger = logger if logger else Logger(verbose, False)
        self.verbose = verbose
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check if the Llama 3.2 Vision model is available in Ollama"""
        try:
            self.logger.info(f"Checking availability of {self.model_name} model in Ollama...")
            
            # Try to list available models to check if our model exists
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            
            if any(self.model_name in model for model in available_models):
                self.logger.info(f"Model {self.model_name} is available in Ollama")
            else:
                self.logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {available_models}")
                self.logger.info(f"Attempting to pull {self.model_name} model...")
                ollama.pull(self.model_name)
                self.logger.info(f"Successfully pulled {self.model_name} model")
                
        except Exception as e:
            self.logger.error(f"Error checking model availability: {e}")
            self.logger.warning("Please ensure Ollama is running and the llama3.2-vision model is available")
            raise
    
    def _read_text_file(self, filepath: str) -> str:
        """Read content from a text file"""
        try:
            self.logger.debug(f"Reading text file: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
            self.logger.info(f"Successfully read text file: {filepath} ({len(content)} characters)")
            return content
        except Exception as e:
            error_msg = f"Error reading text file {filepath}: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _read_pdf_file(self, filepath: str) -> str:
        """Extract text content from a PDF file"""
        try:
            self.logger.debug(f"Reading PDF file: {filepath}")
            text = ""
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                self.logger.debug(f"PDF has {len(pdf_reader.pages)} pages")
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    self.logger.debug(f"Extracted text from page {i+1}: {len(page_text)} characters")
            content = text.strip()
            self.logger.info(f"Successfully read PDF file: {filepath} ({len(content)} characters)")
            return content
        except Exception as e:
            error_msg = f"Error reading PDF file {filepath}: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _read_document(self, filepath: str) -> str:
        """Read content from a document (text or PDF)"""
        file_path = Path(filepath)
        
        if not file_path.exists():
            error_msg = f"File not found: {filepath}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self.logger.info(f"Reading document: {filepath}")
        if file_path.suffix.lower() == '.pdf':
            return self._read_pdf_file(filepath)
        else:
            return self._read_text_file(filepath)
    
    def _generate_response(self, prompt: str, documents: List[str]) -> str:
        """Generate AI response using the prompt and documents via Ollama"""
        self.logger.info("Generating AI response using Ollama...")
        
        # Construct the message content
        combined_content = f"{prompt}\n\nDocuments to process:\n\n"
        for i, doc_content in enumerate(documents, 1):
            combined_content += f"Document {i}:\n{doc_content}\n\n"
        
        self.logger.debug(f"Combined input length: {len(combined_content)} characters")
        
        try:
            self.logger.debug(f"Using Ollama model: {self.model_name}")
            
            # Prepare the message for Ollama
            messages = [{
                'role': 'user',
                'content': combined_content
            }]
            
            # Call Ollama chat API
            response = ollama.chat(
                model=self.model_name,
                messages=messages
            )
            
            # Extract the response content
            if 'message' in response and 'content' in response['message']:
                result = response['message']['content'].strip()
            else:
                result = str(response).strip()
            
            self.logger.info(f"AI response generated successfully ({len(result)} characters)")
            return result
            
        except Exception as e:
            error_msg = f"Error generating response with Ollama: {e}"
            self.logger.error(error_msg)
            return f"{error_msg}\n\nInput was:\n{combined_content}"
    
    def _save_markdown(self, content: str) -> str:
        """Save the AI response to a markdown file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"llm_output_{timestamp}.md"
        output_path = Path(output_filename)
        
        try:
            self.logger.info(f"Saving output to markdown file: {output_filename}")
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(content)
            self.logger.info(f"Successfully saved markdown file: {output_path.absolute()}")
            return str(output_path.absolute())
        except Exception as e:
            error_msg = f"Error saving markdown file: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
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
            error_msg = "At least one file path (prompt file) must be provided"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"Starting document processing with {len(file_paths)} files")
        
        total_steps = 2 + len(file_paths)
        progress = ProgressBar(total_steps, self.verbose)
        
        try:
            prompt_file = file_paths[0]
            document_files = file_paths[1:]
            
            progress.update("Reading prompt file")
            prompt = self._read_document(prompt_file)
            
            documents = []
            for i, doc_file in enumerate(document_files):
                progress.update(f"Reading document {i+1}")
                doc_content = self._read_document(doc_file)
                documents.append(doc_content)
            
            progress.update("Generating AI response")
            response = self._generate_response(prompt, documents)
            
            progress.update("Saving output")
            output_path = self._save_markdown(response)
            
            progress.close()
            self.logger.info(f"Document processing completed successfully")
            
            return output_path
        except Exception as e:
            progress.close()
            self.logger.error(f"Document processing failed: {e}")
            raise


def get_file_paths_interactive(verbose: bool = False, logger: Logger = None) -> List[str]:
    """Get file paths interactively from user input"""
    file_paths = []
    
    # Create interactive IO wrapper if logger provided
    if logger:
        io = InteractiveIO(logger)
    else:
        # Fallback for cases without logger
        class SimpleIO:
            def print(self, msg): print(msg)
            def input(self, prompt): return input(prompt).strip()
        io = SimpleIO()
    
    if verbose:
        io.print("LLM Document Processor")
        io.print("Enter file paths one by one (press Enter with no input to finish):")
    else:
        io.print("LLM Document Processor - Interactive Mode")
        io.print("Enter file paths (empty line to finish):")
    
    while True:
        if not file_paths:
            filepath = io.input("Prompt file: ")
        else:
            filepath = io.input(f"Document {len(file_paths)}: ")
        
        if not filepath:
            break
        
        file_paths.append(filepath)
    
    return file_paths


def process_documents(file_paths: List[str], verbose: bool = False, cleanup_logs: bool = False, logger: Logger = None) -> Tuple[str, bool]:
    """
    Process documents using AI according to a prompt.
    
    Args:
        file_paths: List of file paths where the first is the prompt file
                   and the rest are documents to process
        verbose: Whether to show verbose output
        cleanup_logs: Whether to clean up existing log files
        logger: Existing logger instance to use
    
    Returns:
        Tuple of (output_file_path, success_status)
    """
    try:
        if not logger:
            logger = Logger(verbose, cleanup_logs)
        processor = LLMDocumentProcessor(logger, verbose)
        output_path = processor.process_documents(file_paths)
        return output_path, True
    except Exception as e:
        return str(e), False


def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description='LLM Document Processor')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show verbose output')
    parser.add_argument('--cleanup-logs', action='store_true', help='Clean up existing log files')
    parser.add_argument('files', nargs='*', help='List of files to process (prompt file first)')
    
    args = parser.parse_args()
    
    try:
        # Create single logger instance for the entire runtime
        main_logger = Logger(args.verbose, args.cleanup_logs)
        
        if args.files:
            file_paths = args.files
        else:
            file_paths = get_file_paths_interactive(args.verbose, main_logger)
        
        if not file_paths:
            if not args.verbose:
                print("No files provided. Exiting.")
            return
        
        if not args.verbose:
            print("Processing documents...")
        
        output_path, success = process_documents(file_paths, args.verbose, False, main_logger)  # Don't cleanup again
        
        if success:
            print(f"[SUCCESS] Processing completed successfully")
            if args.verbose:
                print(f"Output saved to: {output_path}")
        else:
            print(f"[ERROR] Processing failed: {output_path}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n[CANCELLED] Operation cancelled by user.")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
# LLM-Doc-Processor

LLM-Doc-Processor is a lightweight tool that transforms prompts and documents into AI-generated outputs. It can run as a standalone program or be integrated into larger systems.

## How It Works
1. Provide a **list of file paths** (all strings).
   - The **first item** must be a text file containing the prompt.
   - The **remaining items** are PDF documents (or text files) to be processed according to the prompt.
2. The program loads a language model, feeds it the prompt and documents, and generates a response.
3. The response is written to a **Markdown file**.
4. The program returns the path of the generated file.

## Features
- Works in standalone mode or as part of a larger pipeline  
- Simple list-based input structure  
- Converts AI output to Markdown for flexible use  

## Example
```python
files = [
    "prompt.txt",          # contains the AI prompt
    "document1.pdf",       # supporting material
    "document2.pdf"        # more supporting material
]

result = process_documents(files)
print("Output saved to:", result)
```
## Output

- AI-generated response saved as a `.md` file  
- Returned path string for easy downstream use  

---

*Built for flexibility: use it on its own, or embed in your own workflows.*

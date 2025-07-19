from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
from transformers.pipelines import pipeline
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import uuid
import re
try:
    import pytesseract
except ImportError:
    pytesseract = None
try:
    from PIL import Image
except ImportError:
    Image = None
import io
from typing import Union  # Added for Python 3.8 compatibility

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://<your-ec2-public-ip>:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize HuggingFace question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# In-memory storage for sessions
sessions = {}

class Question(BaseModel):
    question: str
    session_id: Union[str, None] = None  # Changed from str | None

def preprocess_text(text: str) -> str:
    """Clean and preprocess document text for better QA performance."""
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'[\n\r\t]+', ' ', text)  # Remove excessive newlines/tabs
    return text.strip()

def chunk_text(text: str, max_length: int = 512) -> list:
    """Split text into chunks for processing by the model."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    if pytesseract is None or Image is None:
        raise HTTPException(status_code=500, detail="pytesseract and Pillow must be installed.")
    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        # Read and extract text from PDF
        content = await file.read()
        pdf_document = fitz.open(stream=content, filetype="pdf")
        document_text = ""
        for page in pdf_document:
            # Extract text from page
            try:
                document_text += page.get_text()
            except Exception:
                pass
            # Extract text from images using OCR
            try:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                document_text += pytesseract.image_to_string(img)
            except Exception:
                pass
        pdf_document.close()
        # Preprocess document text
        document_text = preprocess_text(document_text)
        print(f"Extracted text: {document_text[:100]}...")  # Debug log
        # Initialize session
        sessions[session_id] = {
            "document_text": document_text,
            "conversation_history": []
        }
        return {"message": "PDF uploaded and processed successfully", "session_id": session_id, "history": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask")
async def ask_question(question: Question):
    session_id = question.session_id
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid or missing session ID")
    document_text = sessions[session_id]["document_text"]
    conversation_history = sessions[session_id]["conversation_history"]
    if not document_text:
        raise HTTPException(status_code=400, detail="No document uploaded for this session")
    if not question.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        # Build context from conversation history (last 3 entries) and document
        history_context = " ".join(
            f"Q: {item['question']} A: {item['answer']}" 
            for item in conversation_history[-3:]
        )
        full_context = f"{history_context} {document_text}" if history_context else document_text
        # Chunk text and find the best answer
        chunks = chunk_text(full_context)
        best_answer = {"answer": "", "score": 0}
        for chunk in chunks:
            result = qa_pipeline(question=question.question, context=chunk)
            if isinstance(result, dict) and result.get("score", 0) > best_answer["score"]:
                best_answer = result
        # Relaxed threshold and fallback to history
        if best_answer["score"] < 0.05:
            for item in reversed(conversation_history):
                if "certification" in item["question"].lower():
                    answer = item["answer"]
                    break
            else:
                answer = "No relevant answer found in the document."
        else:
            answer = best_answer["answer"]
        # Append to conversation history
        conversation_history.append({
            "question": question.question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        # Limit history size
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        # Update session
        sessions[session_id]["conversation_history"] = conversation_history
        return {"answer": answer, "history": conversation_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
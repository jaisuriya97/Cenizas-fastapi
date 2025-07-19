from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
from transformers import pipeline
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import uuid

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize HuggingFace question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# In-memory storage for sessions (document text and conversation history)
sessions = {}

class Question(BaseModel):
    question: str
    session_id: str | None = None

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
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Read and extract text from PDF
        content = await file.read()
        pdf_document = fitz.open(stream=content, filetype="pdf")
        document_text = ""
        for page in pdf_document:
            document_text += page.get_text()
        pdf_document.close()
        
        # Initialize session with empty history
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
        # Chunk document text and find the best answer
        chunks = chunk_text(document_text)
        best_answer = {"answer": "", "score": 0}
        for chunk in chunks:
            result = qa_pipeline(question=question.question, context=chunk)
            if result["score"] > best_answer["score"]:
                best_answer = result
        
        if best_answer["score"] < 0.1:  # Threshold for irrelevant answers
            answer = "No relevant answer found in the document."
        else:
            answer = best_answer["answer"]
        
        # Append to conversation history with timestamp
        conversation_history.append({
            "question": question.question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit history size to prevent memory issues
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        
        # Update session
        sessions[session_id]["conversation_history"] = conversation_history
        
        return {"answer": answer, "history": conversation_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
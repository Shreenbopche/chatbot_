from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import os
import re
from contextlib import asynccontextmanager

load_dotenv()

# Global variables
client = None
collection = None
data = None

def get_embedding(text: str):
    """Helper function to get embeddings - defined before lifespan"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
# --------- Lifespan context manager ---------
@asynccontextmanager
async def initialize_app(app: FastAPI):
    # Startup: Initialize resources
    global client, collection, data
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Load JSON data
    with open("json_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Setup Chroma with persistence
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="qna_collection", metadata={"hnsw:space":"cosine"})
    
    # Store data if not already present
    if collection.count() == 0:
        for item in data:
            q_eng = item["question"]["english"]
            q_hing = item["question"]["hinglish"]
            q_hin = item["question"]["hindi"]

            for lang, q_text in [("english", q_eng), ("hinglish", q_hing), ("hindi", q_hin)]:
                if q_text.strip():
                    emb = get_embedding(q_text)
                    collection.add(
                        ids=[f"{item['id']}_{lang}"],
                        documents=[q_text],
                        metadatas=[{
                            "answer_english": item["answer"]["english"],
                            "answer_hinglish": item["answer"]["hinglish"],
                            "answer_hindi": item["answer"]["hindi"],
                            "language": lang
                        }],
                        embeddings=[emb]
                    )
        print("✅ Data stored in persistent ChromaDB!")
    else:
        print("✅ ChromaDB already populated!")
    
    yield
    
    # Shutdown: Cleanup (if needed)
    print("🔴 Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="C_First Chatbot",
    description="C_First Chatbot for finance",
    lifespan=initialize_app  
)

# --------- Pydantic Models ---------
class QueryRequest(BaseModel):
    question: str
    similarity_threshold: Optional[float] = 0.7

class QueryResponse(BaseModel):
    status: str
    user_query: str
    bot_response: str
    similarity_score: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    question: str
    database_count: int

#main_function
def rag_answer(user_query: str, similarity_threshold: float = 0.7):
    # Folio number check
    folio_keywords = ["folio", "folio number", "फोलियो", "फोलियो नंबर"]
    numbers = re.findall(r'\d+', user_query)
    for num in numbers:
        if len(num) > 7 and any(keyword.lower() in user_query.lower() for keyword in folio_keywords):
            return "Sorry, I can't provide information about folio numbers.", 0.0

    # Embed user query
    query_emb = get_embedding(user_query)

    # Retrieve top 2 relevant results
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=2,
        include=["documents", "metadatas", "distances"]
    )

    # Handle no results
    if not results["documents"][0]:
        return "No similar question found in the dataset.", 0.0

    top_distance = results["distances"][0][0]
    meta = results["metadatas"][0][0]
    top_doc = results["documents"][0][0]

    print(f"🔍 Top matched question: {top_doc}")
    print(f"🔍 Distance: {top_distance}")
    print(f"🔍 Similarity threshold: {similarity_threshold}")
    print(f"🔍 Checking if {top_distance} <= {1 - similarity_threshold}")

    # ✅ EXACT SAME LOGIC AS WORKING CODE
    if top_distance <= (1 - similarity_threshold):
        lang = meta["language"]
        print(f"✅ Match found! Returning {lang} answer")
        if lang == "english":
            return meta["answer_english"], 1 - top_distance
        elif lang == "hindi":
            return meta["answer_hindi"], 1 - top_distance
        elif lang == "hinglish":
            return meta["answer_hinglish"], 1 - top_distance

    # GPT fallback with finance filter
    print("⚠️ No exact match, checking if question is finance-related...")
    
    # First, check if the question is about finance/stock market
    finance_check_prompt = f"""
You are a financial domain classifier. Determine if the following question is related to finance, stock market, investments, mutual funds, trading, banking, or financial services.

Question: {user_query}

Answer with ONLY "YES" if it's finance-related, or "NO" if it's not finance-related. No explanation needed.
"""

    finance_check = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": finance_check_prompt}],
        temperature=0
    )
    
    is_finance_related = finance_check.choices[0].message.content.strip().upper()
    print(f"🔍 Finance check result: {is_finance_related}")
    
    if is_finance_related != "YES":
        return "I'm sorry, I can only answer questions related to finance, stock market, and investments. Please ask a finance-related question.", 0.0
    
    # If finance-related, proceed with GPT answer using context
    print("✅ Finance-related question, generating answer...")
    contexts = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        contexts.append(f"Q: {doc}\nA (English): {meta['answer_english']}\nA (Hindi): {meta['answer_hindi']}")

    context_text = "\n\n".join(contexts)
    prompt = f"""
You are a helpful financial assistant. Use the following context to answer the user query about finance.

Context:
{context_text}

User Query: {user_query}

Answer in a clear and natural way. Focus only on financial information.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content, 1 - top_distance
# --------- API Endpoints ---------

@app.get("/", response_model=dict)
def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to C_First Chatbot API",
        "endpoints": {
            "POST /chat": "Send a question to the chatbot",
            "GET /docs": "API documentation"
        }
    }

@app.get("/status", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    try:
        db_count = collection.count() if collection else 0
        return HealthResponse(
            status="working",
            message="API is running smoothly",
            database_count=db_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed: {str(e)}")

@app.post("/chat", response_model=QueryResponse)
def chat(query: QueryRequest):
    """
    Main chat endpoint - send a question and get an answer
    """
    try:
        if not query.question or not query.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        answer, similarity = rag_answer(query.question, query.similarity_threshold)
        
        return QueryResponse(
            status="success",
            user_query=query.question,
            bot_response=answer,
            similarity_score=round(similarity, 4) if similarity else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# @app.get("/database/stats")
# def get_database_stats():
#     """Get statistics about the vector database"""
#     try:
#         return {
#             "total_entries": collection.count(),
#             "collection_name": collection.name,
#             "status": "active"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")
from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel
from typing import Union, List
from sentence_transformers import SentenceTransformer
import uvicorn

# ==============================
# CONFIG
# ==============================

import os
API_KEYS = {os.environ.get("API_KEY", "my-secret-key-123")}

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="Custom Embeddings API with Auth")

# ==============================
# AUTH FUNCTION
# ==============================
async def verify_api_key(authorization: str = Header(...)):
    try:
        scheme, token = authorization.split()
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header format")

    if scheme.lower() != "bearer" or token not in API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

    return token

# ==============================
# Schemas
# ==============================
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str], List[int], List[List[int]]]

@app.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    api_key: str = Depends(verify_api_key)
):
    # Normalize input → always a list of strings
    if isinstance(request.input, str):
        texts = [request.input]

    elif isinstance(request.input, list):
        if all(isinstance(x, int) for x in request.input):
            # Single tokenized sequence → turn back into string
            texts = [" ".join(map(str, request.input))]
        elif all(isinstance(x, list) and all(isinstance(i, int) for i in x) for x in request.input):
            # Batch of tokenized sequences
            texts = [" ".join(map(str, x)) for x in request.input]
        elif all(isinstance(x, str) for x in request.input):
            texts = request.input
        else:
            raise HTTPException(status_code=400, detail="Invalid input format")
    else:
        raise HTTPException(status_code=400, detail="Invalid input type")

    # Generate embeddings
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()

    # Construct OpenAI-style response
    data = []
    for i, emb in enumerate(embeddings):
        data.append({
            "object": "embedding",
            "embedding": emb,
            "index": i
        })

    response = {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    }
    return response
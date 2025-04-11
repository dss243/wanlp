from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer
import requests
import os
import secrets
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import pandas as pd
from typing import Optional

# Initialize FastAPI
app = FastAPI()

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    REDDIT_CLIENT_ID = "3aHkvg1zTxdOo3fsekvpnw"
    REDDIT_CLIENT_SECRET = "1va7WqM7ZgtXO9X1eI0f0ePgQtrMjA"
    REDDIT_REDIRECT_URI = "http://localhost:8000/auth/callback"
    REDDIT_AUTH_URL = "https://www.reddit.com/api/v1/authorize"
    REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
    REDDIT_API_URL = "https://oauth.reddit.com"
    SESSION_SECRET = secrets.token_urlsafe(32)

# In-memory session storage (use proper session storage in production)
sessions = {}

# Load Data for Topic Mapping
try:
    df = pd.read_excel(r"C:\Users\Soundous\Desktop\nlp project\nlpproject1\backend\Arabic.xlsx")
    topics = df["Topic"].unique()
    topic_to_id = {topic: i for i, topic in enumerate(topics)}
    id_to_topic = {i: topic for topic, i in topic_to_id.items()}
except Exception as e:
    print(f"Error loading topic mapping: {e}")
    topics = []
    topic_to_id = {}
    id_to_topic = {}

# Model & Tokenizer Setup
try:
    model_name = "aubmindlab/bert-base-arabertv02"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    class MultiTaskModel(nn.Module):
        def __init__(self, model_name, num_topics):
            super(MultiTaskModel, self).__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.hate_speech_classifier = nn.Linear(self.bert.config.hidden_size, 2)
            self.topic_classifier = nn.Linear(self.bert.config.hidden_size, num_topics)

        def forward(self, input_ids, attention_mask):
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = bert_outputs.last_hidden_state[:, 0, :]
            hate_speech_logits = self.hate_speech_classifier(pooled_output)
            topic_logits = self.topic_classifier(pooled_output)
            return hate_speech_logits, topic_logits

    num_topics = len(topics)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(model_name, num_topics).to(device)
    model.load_state_dict(torch.load("arabert_hate_speech_topics (1).pth", map_location=device))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Request Schemas
class TextInput(BaseModel):
    text: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    scope: str
    refresh_token: Optional[str] = None

# Helper Functions
def generate_state() -> str:
    state = secrets.token_urlsafe(16)
    sessions[state] = True  # Store state in session
    return state

def validate_state(state: str) -> bool:
    return state in sessions

def exchange_code_for_token(code: str) -> dict:
    headers = {
        "User-Agent": "ArabicHateSpeechDetector/1.0",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": Config.REDDIT_REDIRECT_URI
    }
    
    try:
        response = requests.post(
            Config.REDDIT_TOKEN_URL,
            data=data,
            auth=(Config.REDDIT_CLIENT_ID, Config.REDDIT_CLIENT_SECRET),
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Token exchange failed: {str(e)}"
        )

def get_reddit_user_info(access_token: str) -> dict:
    headers = {
        "Authorization": f"bearer {access_token}",
        "User-Agent": "ArabicHateSpeechDetector/1.0"
    }
    
    try:
        response = requests.get(
            f"{Config.REDDIT_API_URL}/api/v1/me",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch user info: {str(e)}"
        )

# FastAPI Routes
@app.get("/login/reddit")
async def login_reddit():
    """Initiate Reddit OAuth flow"""
    state = generate_state()
    auth_url = (
        f"{Config.REDDIT_AUTH_URL}?"
        f"client_id={Config.REDDIT_CLIENT_ID}&"
        f"response_type=code&"
        f"state={state}&"
        f"redirect_uri={Config.REDDIT_REDIRECT_URI}&"
        f"duration=permanent&"
        f"scope=read identity"
    )
    return RedirectResponse(url=auth_url)

from urllib.parse import urlencode
from fastapi.responses import RedirectResponse

@app.get("/auth/callback")
async def reddit_callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None
):
    """Handle Reddit OAuth callback and redirect to frontend dashboard with user info"""
    if error:
        raise HTTPException(
            status_code=400,
            detail=f"Reddit authorization failed: {error}"
        )
    
    if not code or not state:
        raise HTTPException(
            status_code=400,
            detail="Missing code or state parameter"
        )
    
    if not validate_state(state):
        raise HTTPException(
            status_code=400,
            detail="Invalid state parameter"
        )
    
    try:
        # Exchange code for token
        token_data = exchange_code_for_token(code)
        
        # Get user info from Reddit
        user_data = get_reddit_user_info(token_data["access_token"])
        
        # Build redirect URL to React Dashboard
        params = {
            "access_token": token_data["access_token"],
            "user_name": user_data.get("name", "reddit_user"),
            "user_id": user_data.get("id"),
            "icon_img": user_data.get("icon_img", "")
        }
        query_string = urlencode(params)
        redirect_url = f"http://localhost:3000/dashboard?{query_string}"
        
        return RedirectResponse(url=redirect_url)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Authentication failed: {str(e)}"
        )


@app.post("/predict/")
async def predict_endpoint(data: TextInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        inputs = tokenizer(
            data.text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            hate_speech_logits, topic_logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
        hate_pred = torch.argmax(hate_speech_logits, dim=1).item()
        topic_pred = torch.argmax(topic_logits, dim=1).item()
        
        return {
            "text": data.text,
            "hate_speech": "Hate Speech" if hate_pred == 1 else "Not Hate Speech",
            "topic": id_to_topic.get(topic_pred, "Unknown"),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/analyze-me/")
async def analyze_reddit_user(
    request: Request,
    limit: int = 10
):
    """Analyze authenticated user's Reddit content"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid authorization header"
        )
    
    access_token = auth_header.split(" ")[1]
    
    try:
        # Get user's recent posts and comments
        headers = {
            "Authorization": f"bearer {access_token}",
            "User-Agent": "ArabicHateSpeechDetector/1.0"
        }
        
        # Get recent posts
        posts_response = requests.get(
            f"{Config.REDDIT_API_URL}/user/me/submitted",
            headers=headers,
            params={"limit": limit}
        )
        posts_response.raise_for_status()
        
        # Get recent comments
        comments_response = requests.get(
            f"{Config.REDDIT_API_URL}/user/me/comments",
            headers=headers,
            params={"limit": limit}
        )
        comments_response.raise_for_status()
        
        # Process content
        content = []
        for item in posts_response.json().get("data", {}).get("children", []):
            content.append(item["data"].get("selftext", ""))
        
        for item in comments_response.json().get("data", {}).get("children", []):
            content.append(item["data"].get("body", ""))
        
        # Analyze content
        results = []
        hate_speech_count = 0
        
        for text in content:
            if text.strip():
                try:
                    prediction = predict(text)
                    results.append({
                        "text": text,
                        "hate_speech": prediction["hate_speech"],
                        "topic": prediction["topic"]
                    })
                    if prediction["hate_speech"] == "Hate Speech":
                        hate_speech_count += 1
                except:
                    continue
        
        # Calculate statistics
        total_analyzed = len(results)
        hate_speech_percentage = (hate_speech_count / total_analyzed * 100) if total_analyzed > 0 else 0
        
        # Count topics
        topics_distribution = {}
        for result in results:
            topic = result["topic"]
            topics_distribution[topic] = topics_distribution.get(topic, 0) + 1
        
        return {
            "total_posts": len(posts_response.json().get("data", {}).get("children", [])),
            "total_comments": len(comments_response.json().get("data", {}).get("children", [])),
            "total_analyzed": total_analyzed,
            "hate_speech_count": hate_speech_count,
            "hate_speech_percentage": hate_speech_percentage,
            "topics_distribution": topics_distribution,
            "sample_results": results[:5]  # Return first 5 results as sample
        }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch Reddit content: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "Arabic Hate Speech & Topic Detection API is live."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
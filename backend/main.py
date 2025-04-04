from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import pandas as pd

# Initialize FastAPI
app = FastAPI()

# Enable CORS for all origins (Frontend compatibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Load Data for Topic Mapping ====
df = pd.read_excel(r"C:\Users\Soundous\Desktop\nlp project\nlpproject1\backend\Arabic.xlsx")  # Make sure this file exists in backend folder
topics = df["Topic"].unique()
topic_to_id = {topic: i for i, topic in enumerate(topics)}
id_to_topic = {i: topic for topic, i in topic_to_id.items()}

# ==== Model & Tokenizer Setup ====
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

# Load trained model
num_topics = len(topics)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskModel(model_name, num_topics).to(device)
model.load_state_dict(torch.load("arabert_hate_speech_topics (1).pth", map_location=device))
model.eval()

# ==== Request Schema ====
class TextInput(BaseModel):
    text: str

# ==== Prediction Logic ====
def predict(text: str):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        hate_speech_logits, topic_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    hate_pred = torch.argmax(hate_speech_logits, dim=1).item()
    topic_pred = torch.argmax(topic_logits, dim=1).item()
    return {
        "text": text,
        "hate_speech": "Hate Speech" if hate_pred == 1 else "Not Hate Speech",
        "topic": id_to_topic[topic_pred]
    }

# ==== FastAPI Routes ====
@app.post("/predict/")
async def predict_endpoint(data: TextInput):
    return predict(data.text)

@app.get("/")
async def root():
    return {"message": "Arabic Hate Speech & Topic Detection API is live."}

# ==== Run App ====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import google.generativeai as genai
from langchain.cache import InMemoryCache
import sqlite3
from datetime import datetime
import langchain

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "user_histories.db")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
BASE_URL = "https://graph.facebook.com/v21.0"

import os
import os
import requests

# Define the GitHub URL and the local file path
GITHUB_URL = "https://raw.githubusercontent.com/advik-7/IG/main/Company%20Text%20database.pdf"
LOCAL_FILE_PATH = "Company_Text_database.pdf"

# Check if the file exists locally; if not, download it
if not os.path.exists(LOCAL_FILE_PATH):
    print("Downloading file from GitHub...")
    response = requests.get(GITHUB_URL)
    if response.status_code == 200:
        with open(LOCAL_FILE_PATH, "wb") as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved as {LOCAL_FILE_PATH}")
    else:
        print("Failed to download the file.")
        exit(1)

# Proceed with using the local file
file_path = LOCAL_FILE_PATH


import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import google.generativeai as genai
from langchain.cache import InMemoryCache
from typing import Optional, List, Mapping, Any
import sqlite3
from datetime import datetime
# Use in-memory caching instead of Redis
import langchain
langchain.cache = InMemoryCache()

class SalesAgentBot:
    def __init__(self, file_path: str, db_path: str, batch_size: int = 5):
        self.file_path = file_path
        self.history = []
        self.batch_size = batch_size
        self.db_path = db_path
        self._initialize_database()

        # Load documents and prepare embeddings
        self.loader = PyPDFLoader(file_path)
        self.documents = self.loader.load()

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = self.text_splitter.split_documents(self.documents)

        # Load and quantize the Sentence Transformer model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Precompute and store embeddings
        self.embeddings = self.model.encode([t.page_content for t in self.texts], show_progress_bar=True, device='cpu')

        class CustomEmbeddings:
            def __init__(self, model):
                self.model = model

            def embed_documents(self, texts):
                return self.model.encode(texts, batch_size=self.batch_size)

            def embed_query(self, text):
                return self.model.encode([text])[0]

            def __call__(self, text):
                return self.model.encode([text])[0]

        self.custom_embeddings = CustomEmbeddings(self.model)

        # Set up FAISS for document retrieval
        self.db = FAISS.from_embeddings(
            text_embeddings=[(t.page_content, embedding) for t, embedding in zip(self.texts, self.embeddings)],
            embedding=self.custom_embeddings
        )

        # Google Gemini API configuration
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 512,
        }

        self.model_instance = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        # Custom wrapper for interacting with Google Gemini
        class CustomGemini:
            def __init__(self, temperature: float, max_tokens: int, model: str, google_api_key: str):
                self.temperature = temperature
                self.max_tokens = max_tokens
                self.model = model
                genai.configure(api_key=google_api_key)
                self.generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                self.model_instance = genai.GenerativeModel(
                    model_name=model,
                    generation_config=self.generation_config,
                )

            def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                response = self.model_instance.generate_content(prompt)
                return response.text

        self.custom_gemini = CustomGemini(
            temperature=0.8,
            max_tokens=100,
            model="gemini-1.5-flash",
            google_api_key=os.environ["GEMINI_API_KEY"]
        )

        class CustomLLMWrapper(LLM):
            custom_llm: CustomGemini

            @property
            def _llm_type(self) -> str:
                return "custom_gemini"

            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                return self.custom_llm(prompt, stop=stop)

            @property
            def _identifying_params(self) -> Mapping[str, Any]:
                return {
                    "temperature": self.custom_llm.temperature,
                    "max_tokens": self.custom_llm.max_tokens,
                    "model": self.custom_llm.model,
                }

        self.wrapped_llm = CustomLLMWrapper(custom_llm=self.custom_gemini)

        # Set up the retriever and QA chain
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})
        self.qa = RetrievalQA.from_chain_type(
            llm=self.wrapped_llm, chain_type="stuff", retriever=self.retriever
        )

    def _initialize_database(self):
        """Create the user_histories table if it doesn't exist."""
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_histories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
        connection.close()

    def save_user_message(self, user_id: str, message: str):
        """Save a message to the database."""
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        cursor.execute("INSERT INTO user_histories (user_id, message) VALUES (?, ?)", (user_id, message))
        connection.commit()
        connection.close()

    def get_user_history(self, user_id: str) -> List[str]:
        """Retrieve conversation history for a user."""
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        cursor.execute("SELECT message FROM user_histories WHERE user_id = ? ORDER BY timestamp ASC", (user_id,))
        rows = cursor.fetchall()
        connection.close()
        return [row[0] for row in rows]

    def process_query(self, user_id: str, query: str) -> str:
        """Process user query and generate a response."""
        self.save_user_message(user_id, f"User: {query}")
        history = self.get_user_history(user_id)

        # Get the most recent 4-5 messages from the conversation history
        conversation_context = "\n".join(history[-5:])
        prompt = f"""
        You are a friendly and professional sales agent. Answer the customer's question briefly while maintaining clarity and helpfulness, acting as a company representative.

        Here is the conversation so far:
        {conversation_context}

        Here is the customer's question: "{query}"
        Based on the company's info, provide a helpful response.
        """
        response = self.qa.run(prompt)
        self.save_user_message(user_id, f"SalesBot: {response}")
        return response

import os
db_path = "user_histories.db"

bot = SalesAgentBot(file_path, db_path)
import re
from datetime import datetime, timedelta

def chatbot_response(user_id: str, user_message: str):
    if not user_message:
        return "Server Issue :)"
    else:
        # Process the query for the specific user
        response = bot.process_query(user_id, user_message)
        return response

from flask import Flask, request, jsonify
import hashlib
import hmac
import time
import requests
import requests
import os

BASE_URL = "https://graph.facebook.com/v21.0"

def get_pages():
    endpoint = f"{BASE_URL}/me/accounts"
    params = {"access_token": ACCESS_TOKEN}
    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        pages_data = response.json()
        pages_info = []

        for page in pages_data.get('data', []):
            page_id = page.get('id')
            page_access_token = page.get('access_token')
            pages_info.append({
                "page_id": page_id,
                "access_token": page_access_token
            })

        return pages_info
    else:
        print("Error occurred:")
        print(response.json())
        return None



output=get_pages()
PAGE_ACCESS_TOKEN = output[0]['access_token']
VERIFY_TOKEN = "nigga"
APP_SECRET = output[0]['access_token']
from fastapi import FastAPI, Request, Depends, HTTPException ,Query
from pydantic import BaseModel
import requests
import hmac
import uvicorn
from fastapi import FastAPI, Request, Query, HTTPException, Header
from fastapi.responses import PlainTextResponse
import httpx
import hashlib
import hmac
import asyncio

app = FastAPI()

# Global Counter for Gemini API Calls
gemini_request_count = 0  

@app.get("/")
async def home():
    return {"message": "Welcome to My App!"}

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(..., alias="hub.mode"),
    hub_challenge: int = Query(..., alias="hub.challenge"),
    hub_verify_token: str = Query(..., alias="hub.verify_token"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        print("WEBHOOK_VERIFIED")
        return PlainTextResponse(str(hub_challenge), status_code=200)
    raise HTTPException(status_code=403, detail="Forbidden")
from fastapi import FastAPI, Request, HTTPException
import time



# In-memory cache for deduplication
recent_message_ids = {}

# Expiry time for deduplication (e.g., 10 minutes)
EXPIRY_TIME = 120 

@app.post("/webhook")
async def handle_webhook(request: Request):
    global gemini_request_count  
    body = await request.json()
    current_time = time.time()

    if body.get("object") == "instagram":
        for entry in body.get("entry", []):
            for event in entry.get("messaging", []):
                sender_id = event.get("sender", {}).get("id")
                message = event.get("message", {})
                message_id = message.get("mid")  # Unique message ID from Instagram

                if sender_id and message and message_id:
                    # Deduplication check
                    if message_id in recent_message_ids and current_time - recent_message_ids[message_id] < EXPIRY_TIME:
                        return {"status": "DUPLICATE_IGNORED"}

                    # Store the message ID
                    recent_message_ids[message_id] = current_time

                    # Cleanup old entries
                    for key in list(recent_message_ids.keys()):
                        if current_time - recent_message_ids[key] > EXPIRY_TIME:
                            del recent_message_ids[key]

                    message_text = message.get("text", "")
                    is_deleted = message.get("is_deleted", False)
                    is_echo = message.get("is_echo", False)

                    if not is_echo and not is_deleted:
                        print(f"Instagram Message from {sender_id}: {message_text}")

                        # Track Gemini API request
                        gemini_request_count += 1
                        print(f"Total Gemini Requests Made: {gemini_request_count}")

                        response_text = chatbot_response(sender_id, message_text)

                        # Send the response back
                        try:
                            response = await send_message(sender_id, response_text)
                            print("Reply sent successfully:", response)
                        except Exception as error:
                            print("Failed to send reply:", error)

        return {"status": "EVENT_RECEIVED"}

    elif body.get("object") == "page":
        for entry in body.get("entry", []):
            messaging_events = entry.get("messaging", [])
            if not messaging_events:
                continue  # Skip if no messaging events

            webhook_event = messaging_events[0]
            print("Incoming Webhook Event: ", webhook_event)

            if "message" in webhook_event and "text" in webhook_event["message"]:
                sender_id = webhook_event["sender"]["id"]
                message_text = webhook_event["message"]["text"]

                # Log the incoming message and send a reply
                await print_message(sender_id, message_text)

        return {"status": "EVENT_RECEIVED"}

    raise HTTPException(status_code=404, detail="Not Found")  

async def print_message(sender_id: str, message_text: str):
    global gemini_request_count  
    print(f"Message from {sender_id}: {message_text}")

    # Track Gemini API request
    gemini_request_count += 1
    print(f"Total Gemini Requests Made: {gemini_request_count}")

    response_text = chatbot_response(sender_id, message_text)
    print(response_text)

    try:
        response = await send_message(sender_id, response_text)
        print("Reply sent successfully:", response)
    except Exception as error:
        print("Failed to send reply:", error)

async def send_message(ssid: str, text: str):
    url = "https://graph.facebook.com/v21.0/me/messages"
    payload = {
        "recipient": {"id": ssid},
        "message": {"text": text},
    }
    headers = {"Content-Type": "application/json"}
    params = {"access_token": PAGE_ACCESS_TOKEN}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers, params=params)
        response.raise_for_status()  # Raise error if request fails
        return response.json()

async def validate_signature(req: Request, x_hub_signature_256: str = Header(None)):
    if not x_hub_signature_256:
        print("No signature found in headers.")
        return False

    method, signature_hash = x_hub_signature_256.split("=")
    if method != "sha256":
        print("Unknown signature method.")
        return False

    body = await req.body()
    expected_hash = hmac.new(
        APP_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()

    return signature_hash == expected_hash

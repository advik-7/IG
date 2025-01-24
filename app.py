from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import google.generativeai as genai


class SalesAgentBot:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.history = []

        # Load documents and prepare embeddings
        self.loader = PyPDFLoader(file_path)
        self.documents = self.loader.load()

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = self.text_splitter.split_documents(self.documents)

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode([t.page_content for t in self.texts])

        class CustomEmbeddings:
            def __init__(self, model):
                self.model = model

            def embed_documents(self, texts):
                return self.model.encode(texts)

            def embed_query(self, text):
                return self.model.encode(text)

            def __call__(self, text):
                return self.model.encode(text)

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
            temperature=0.7,
            max_tokens=512,
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

    def process_query(self, query: str) -> str:
        # Add the user's query to the conversation history
        self.history.append(f"User: {query}")

        # Add prompt for acting as a sales agent
        prompt = f"""
        You are a friendly and professional sales agent. Answer the customer's question briefly and concisely while maintaining helpfulness and clarity. Keep your responses short and to the point.

        If the customer wishes to book a slot, ask them to type: "I wish to book a slot [date] [time]", replacing [date] and [time] with their preferred booking details. For example: "I wish to book a slot 2025-01-15 14:30".

        Here is the customer's question: "{query}"
        Based on the company's information, provide a concise and helpful response.
        """


        # Get the response from the model
        response = self.qa.run(prompt)

        # Add the model's response to the conversation history
        self.history.append(f"SalesBot: {response}")

        # Return the response to the user
        return response

    def get_conversation_history(self) -> str:
        # Return the entire conversation history
        return "\n".join(self.history)

import os
os.environ["GEMINI_API_KEY"]="AIzaSyCF2Xymk8vra8xjTh3QIIEfrLoXRIHMmLk"

file_path="company_profile_practise1.pdf"


bot = SalesAgentBot(file_path)
import re
from datetime import datetime, timedelta

# Function to handle chatbot processing
def chatbot_response(user_message: str):
    # Check if the user is making a booking request
    if check_booking_request(user_message):
        # If it's a valid booking request, the function already handled it
        return "appoinment booked :)"
    else:
        # Otherwise, process the query normally
        response = bot.process_query(user_message)
        return response




from flask import Flask, request, jsonify
import hashlib
import hmac
import time
import requests

import requests
import os

ACCESS_TOKEN = "EAAY4S7hHfy4BO7kOk0UJWlQMwNUxVZC1axqeEyEfdPVlHqYRYs4xvZCICUew4dCsmzelvyj6AeVZCx9fLYKZBOAAyYAaTvNpeYfncazls1FqJkY4Fh10G2dina4ZBRRlRSxV8eprpfNmBqMq1r5FGFPfxKCerrRgwZAPLmQ8YmXWZCCJdZAQ0e3ZCu3MW"
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
app = Flask(__name__)

# Webhook verification
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            print("WEBHOOK_VERIFIED")
            return challenge, 200
        else:
            return "Forbidden", 403

# Handling incoming messages asynchronously
@app.route("/webhook", methods=["POST"])
def handle_webhook():
    body = request.get_json()

    if body.get("object") == "instagram":
        for entry in body.get("entry", []):
            messaging_events = entry.get("messaging", [])
            for event in messaging_events:
                sender_id = event.get("sender", {}).get("id")
                message = event.get("message", {})

                if sender_id and message:
                    message_text = message.get("text", "")
                    is_deleted = message.get("is_deleted", False)
                    is_echo = message.get("is_echo", False)

                    # Only process non-deleted and non-echo messages
                    if not is_echo and not is_deleted:
                        print(f"Instagram Message from {sender_id}: {message_text}")
                        
                        # Generate a response
                        response_text = chatbot_response(message_text)

                        # Send the response back
                        try:
                            response = send_message(sender_id, response_text)
                            print("Reply sent successfully:", response)
                        except Exception as error:
                            print("Failed to send reply:", error)
        
        return "EVENT_RECEIVED", 200
 
    # Validate the event type
    elif body.get("object") == "page":
        for entry in body.get("entry", []):
            webhook_event = entry.get("messaging", [])[0]
            print("Incoming Webhook Event: ", webhook_event)

            # Log the message text if available
            if "message" in webhook_event and "text" in webhook_event["message"]:
                sender_id = webhook_event["sender"]["id"]
                message_text = webhook_event["message"]["text"]

                # Log the incoming message and send a reply
                print_message(sender_id, message_text)

        # Respond to Facebook server
        return "EVENT_RECEIVED", 200
    else:
        return "Not Found", 404

def print_message(sender_id, message_text):
    print(f"Message from {sender_id}: {message_text}")
    
    response_text = chatbot_response(message_text)

    try:
        response = send_message(sender_id, response_text)
        print("Reply sent successfully:", response)
    except Exception as error:
        print("Failed to send reply:", error)

    # Return the input for further use
    return sender_id, message_text, response_text

# Function to send a message using Facebook Messenger API
def send_message(ssid, text):
    endpoint = "https://graph.facebook.com/v21.0/me/messages"
    payload = {
        "recipient": {"id": ssid},
        "message": {"text": text},
    }
    headers = {
        "Content-Type": "application/json",
    }
    params = {
        "access_token": PAGE_ACCESS_TOKEN,
    }

    response = requests.post(endpoint, json=payload, headers=headers, params=params)
    if response.status_code == 200:
        print(f"Message sent to {ssid}: {text}")
        return response.json()  # API response
    else:
        print("Failed to send message:", response.text)
        response.raise_for_status()

# Define your Agent function here
def Agent(message_text):
    
    response_text = bot.process_query(message_text)
    return response_text



# Signature validation (optional but recommended)
def validate_signature(req):
    signature = req.headers.get("x-hub-signature-256")
    if not signature:
        print("No signature found in headers.")
        return False

    method, signature_hash = signature.split("=")
    if method != "sha256":
        print("Unknown signature method.")
        return False

    expected_hash = hmac.new(
        APP_SECRET.encode(),
        req.data,
        hashlib.sha256
    ).hexdigest()

    return signature_hash == expected_hash

# Start server
if __name__ == "__main__":
    app.run()

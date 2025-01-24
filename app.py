# Import required libraries
from cohere import Client
import logging, json
from pymongo import MongoClient
import streamlit as st
import os
import certifi
import numpy as np

# Function to establish MongoDB connection
def get_database_connection():
   try:
       uri = "mongodb+srv://insansabddp:3SqXcP41pHBjR0nH@cluster0.zz4x9.mongodb.net/GovernmentPortal"
       client = MongoClient(uri, tlsCAFile=certifi.where())
       client.admin.command('ping')
       return client.GovernmentPortal
   except Exception as e:
       logging.error(f"DB Error: {str(e)}")
       return None

# Client class to handle Cohere API interactions
class CohereAPIClient:
   def __init__(self):
       # Initialize Cohere client with API key
       self.api_key = os.getenv("COHERE_API_KEY", "rNjBbhgU0z3qmSmnvAo05Duc95z6nWkOdf9iFNna")
       self.client = Client(api_key=self.api_key)

   # Get text embeddings for semantic search
   def get_embeddings(self, texts):
       try:
           response = self.client.embed(
               texts=texts,
               model='embed-english-v3.0',
               input_type='search_query'
           )
           return response.embeddings
       except Exception as e:
           logging.error(f"Embedding error: {str(e)}")
           return [[0] * 1024] * len(texts)

   # Generate response using Cohere LLM
   def generate_response(self, query, exam_data):
       try:
           # Construct prompt with instructions
           prompt = f"""You are an exam information assistant. Query: {query}

           Instructions:
           - Only state information explicitly present in the data
           - Do not make assumptions or add details not in data
           - If a field is missing, state "No information available for [field]"
           - Format response clearly with bullet points
           - Mark tentative/unconfirmed information

           Exam Data: {json.dumps(exam_data, ensure_ascii=False)}"""

           response = self.client.generate(
               prompt=prompt,
               max_tokens=500,
               temperature=0.3,
               model='command'
           )
           return response.generations[0].text
       except Exception as e:
           logging.error(f"Generation error: {str(e)}")
           return str(e)

# Main bot class to handle exam queries
class ExamBot:
   def __init__(self):
       # Initialize database and Cohere client
       self.db = get_database_connection()
       self.llm = CohereAPIClient()

   # Search exams in database
   def search_exams(self, query: str):
       try:
           # Query MongoDB with expanded fields
           exams = list(self.db.events.find(
               {"event_type": "Exam"},
               {
                   'name': 1,
                   'date_of_notification': 1, 
                   'end_date': 1,
                   'details.eligibility': 1,
                   'details.application_fee': 1,
                   'details.important_dates': 1,
                   'details.selection_process': 1,
                   'details.salary': 1,
                   'details.vacancies': 1
               }
           ).limit(10))

           return exams
           
       except Exception as e:
           logging.error(f"Search error: {str(e)}")
           return []

   # Process query and generate response
   def get_response(self, query: str) -> str:
       try:
           # Get relevant exams
           relevant_exams = self.search_exams(query)
           if not relevant_exams:
               return "No exam information available."

           # Format exam details
           exam_details = []
           for e in relevant_exams:
               if not isinstance(e, dict):
                   continue
                   
               details = e.get('details', {})
               if isinstance(details, str):
                   details = {}

               exam = {
                   "name": str(e.get("name", "")),
                   "notification_date": str(e.get("date_of_notification", "")),
                   "end_date": str(e.get("end_date", "")),
                   "details": details
               }
               exam_details.append(exam)

           # Generate response using Cohere
           return self.llm.generate_response(query, exam_details)

       except Exception as e:
           logging.error(f"Error: {str(e)}")
           return f"Error: {str(e)}"

# Streamlit UI setup
def main():
   st.title("Exam Information Chatbot")

   # Initialize session state for chat history
   if "messages" not in st.session_state:
       st.session_state.messages = []

   bot = ExamBot()

   # Display chat history
   for message in st.session_state.messages:
       with st.chat_message(message["role"]):
           st.write(message["content"])

   # Handle new user input
   if prompt := st.chat_input("Ask about exams..."):
       st.session_state.messages.append({"role": "user", "content": prompt})
       with st.chat_message("user"):
           st.write(prompt)

       response = bot.get_response(prompt)
       st.session_state.messages.append({"role": "assistant", "content": response})
       with st.chat_message("assistant"):
           st.write(response)

if __name__ == "__main__":
   main()
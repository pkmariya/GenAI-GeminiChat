# Simple chat app using Langchain and Gemini Model

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Load the environment variables
load_dotenv()

# Instantiate model object and generate chat completion:
llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-pro', 
    temperature=0,
    max_tokens=None, 
    timeout=None, 
    max_retries=2,
)

# Create the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a chatbot"),
        ("human", "Question: {question}")
    ]
)

st.title("Chat App using LangChain and Gemini LLM!")
input_text = st.text_input("Enter your question here")

output_parser = StrOutputParser()

# Create the chain
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
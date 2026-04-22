import streamlit as st
import os
import tempfile
from dotenv import load_dotenv # Make sure to pip install python-dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file (Grabs GOOGLE_API_KEY automatically)
load_dotenv()

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Finance Tutor", page_icon="📈", layout="wide")
st.title("📈 AI Finance Quiz & Tutor Bot")
st.markdown("Upload a financial report, and I will test your knowledge on it!")

# --- 2. Session State Initialization ---
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "current_context" not in st.session_state:
    st.session_state.current_context = ""
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. Sidebar: Configuration & Upload ---
with st.sidebar:
    st.header("⚙️ Setup")
    uploaded_file = st.file_uploader("Upload Investment Report (PDF)", type="pdf")
    
    if st.button("Process Document"):
        if not uploaded_file:
            st.error("Please upload a PDF document first.")
        else:
            with st.spinner("Processing document and building database..."):
                # Save the uploaded file temporarily so LangChain can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
                
                # Ingestion & Chunking
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                
                # Create In-Memory Vector Database
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
                
                st.success("Database built successfully! You can now generate a quiz.")

# --- 4. Main UI: Quiz Generation & Chat ---
if st.session_state.vector_db is not None:
    
    # Button to generate a new question
    if st.button("Generate New Quiz Question"):
        with st.spinner("Reading the report and thinking of a question..."):
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
             
            # Retrieve random context
            docs = st.session_state.vector_db.similarity_search("investment risks returns market", k=1)
            st.session_state.current_context = docs[0].page_content
            
            # Using triple quotes for multi-line explicit formatting
            question_prompt = PromptTemplate(
                input_variables=["context"],
                template="""Based on the following financial context, generate a single, challenging question for a finance student. 

If the context is not related to finance, state that and ask for a financial document. Do NOT provide the answer.



Context: {context}
Question:"""
            )
        
            question_chain = question_prompt | llm
            quiz_question = question_chain.invoke({"context": st.session_state.current_context}).content
            
            # Save to session state
            st.session_state.current_question = quiz_question
            
            # Add to messages (using \n\n to ensure Streamlit pushes text to a new line)
            st.session_state.messages.append({"role": "assistant", "content": f"**New Question:**\n\n{quiz_question}"})

# --- 5. Render Chat History ---
# IMPORTANT: This block is now unindented so it runs on every page refresh!
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. Chat Input & Evaluation ---
# If a question has been generated, allow the user to answer
if st.session_state.current_question:
    if user_answer := st.chat_input("Type your answer here..."):
        # Display user's answer
        st.session_state.messages.append({"role": "user", "content": user_answer})
        with st.chat_message("user"):
            st.markdown(user_answer)
            
        # Evaluate the answer
        with st.chat_message("assistant"):
            with st.spinner("Evaluating your answer..."):
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                eval_prompt = PromptTemplate(
                    input_variables=["context", "question", "user_answer"],
                    template="""
                    You are an expert Finance Tutor. 
                    Context from Investment Report: {context}
                    Question asked: {question}
                    Student's Answer: {user_answer}
                    
                    Evaluate the student's answer. If it is correct, confirm it. 
                    If it is incorrect or incomplete, explain why using ONLY the facts from the Context provided. 
                    Be encouraging but academically rigorous. Format your response cleanly.
                    """
                )
                eval_chain = eval_prompt | llm
                feedback = eval_chain.invoke({
                    "context": st.session_state.current_context, 
                    "question": st.session_state.current_question, 
                    "user_answer": user_answer
                }).content
                
                st.markdown(feedback)
                st.session_state.messages.append({"role": "assistant", "content": feedback})
                
                # Clear current question so the user has to generate a new one
                st.session_state.current_question = ""
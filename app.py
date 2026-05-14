import streamlit as st
import os
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# --- Page Config ---
st.set_page_config(page_title="Mental Health RAG Bot", page_icon="🧠", layout="wide")

# --- UI Sidebar (Experimentation) ---
st.sidebar.title("🛠️ RAG Parameters")
st.sidebar.markdown("Experiment with these settings to see how the chatbot responds.")

k_docs = st.sidebar.slider("Number of Retrieved Documents (k)", min_value=1, max_value=5, value=2)
chunk_size = st.sidebar.slider("Chunk Size (Characters)", min_value=100, max_value=1000, value=500, step=100)
temperature = st.sidebar.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# --- Caching Expensive Operations ---
@st.cache_resource(show_spinner="Loading Local LLM (This takes a few minutes on first run to download model)...")
def load_llm(temp):
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load model. Automatically uses GPU if available.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # We use a trick: temperature = 0.0 raises an error in some transformers versions, so we use a very small float
    temp_val = temp if temp > 0 else 0.01
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=temp_val,
        do_sample=temp_val > 0,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource(show_spinner="Loading Embeddings...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Processing Dataset and building Vector Store...")
def build_vector_store(_embeddings, chunk_sz):
    file_path = "Mental_Health_FAQ.csv"
    if not os.path.exists(file_path):
        st.error(f"Dataset not found at {file_path}. Please ensure the file is in the same directory.")
        st.stop()
        
    loader = CSVLoader(file_path=file_path, encoding='utf-8', source_column="Questions")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_sz, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(texts, _embeddings)
    return vectorstore

# --- Load Models & Data ---
try:
    llm = load_llm(temperature)
    embeddings = load_embeddings()
    vectorstore = build_vector_store(embeddings, chunk_size)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_docs})
except Exception as e:
    st.error(f"An error occurred during setup: {e}")
    st.stop()

# --- Custom Prompt Template for Qwen ---
prompt_template = """<|im_start|>system
You are a helpful and compassionate mental health assistant. Use the following context to answer the user's question. If you don't know the answer based on the context, just say you don't know, don't try to make up an answer. Keep your answer concise.
Context:
{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# --- Define LCEL RAG Chain ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# --- Chat Interface ---
st.title("🧠 Mental Health Assistant (Local RAG)")
st.markdown("Ask me anything about mental health based on the FAQ dataset! (Runs 100% locally on your machine)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if user_input := st.chat_input("E.g., What are the signs of mental illness?"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # First, get the retrieved documents manually for display
                docs = retriever.invoke(user_input)
                
                # Generate the response using our LCEL chain
                full_text = qa_chain.invoke(user_input)
                
                # Qwen might return the prompt along with the answer, so we extract just the assistant part
                if "<|im_start|>assistant" in full_text:
                    answer = full_text.split("<|im_start|>assistant")[-1].strip()
                else:
                    answer = full_text.strip()
                
                # Remove any trailing end tokens
                answer = answer.replace("<|im_end|>", "").strip()
                
                st.markdown(answer)
                
                # Show sources to fulfill the "retriever" transparency
                with st.expander("Show Retrieved Sources"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Source {i+1}:** {doc.page_content[:200]}...")
                        
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating response: {e}")

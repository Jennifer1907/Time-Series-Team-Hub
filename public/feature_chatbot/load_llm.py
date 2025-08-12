import streamlit as st #? run app streamlit run file_name.py
import tempfile
import os
import torch


from transformers import(
						AutoTokenizer, # Tokenize Model
						AutoModelForCausalLM,  # LLM Loader - used for loading and using pre-trained models designed for causal language modeling tasks
						)
from transformers.pipelines import pipeline # pipeline to setup llm-task oritented model
from langchain_huggingface import HuggingFaceEmbeddings # huggingface sentence_transformer embedding models
from langchain_huggingface.llms import HuggingFacePipeline # like transformer pipeline

from langchain.memory import ConversationBufferMemory # Deprecated
from langchain_community.chat_message_histories import ChatMessageHistory # Deprecated
from langchain_community.document_loaders import PyPDFLoader, TextLoader # PDF Processing
from langchain.chains import ConversationalRetrievalChain # Deprecated
from langchain_experimental.text_splitter import SemanticChunker # module for chunking text

from langchain_text_splitters import RecursiveCharacterTextSplitter # recursively divide text, then merge them together if merge_size < chunk_size
from langchain_core.runnables import RunnablePassthrough # Use for testing (make 'example' easy to execute and experiment with)
from langchain_core.output_parsers import StrOutputParser # format LLM's output text into (list, dict or any custom structure we can work with)
from langchain import hub
from langchain_core.prompts import PromptTemplate
import json


#? Read huggingface token in token.txt file. Please paste your huggingface token in token.txt
@st.cache_resource
def get_hg_token():
    #! RESET huggingface token if get TOKEN ERROR
    with open('token.txt', 'r') as f:
        hg_token = f.read()
        return hg_token


@st.cache_resource
def load_embeddings():
    """Tải mô hình embedding tiếng Việt"""
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

# Save downloaded LLM
if 'llm' not in st.session_state:
    st.session_state.llm = None



@st.cache_resource
def load_llm():
    MODEL_NAME = "google/gemma-2-2b-it"  # hoặc mô hình nhỏ khác

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32  # chạy được trên CPU
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_pipeline = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        device=-1  # CPU
    )

    return HuggingFacePipeline(pipeline=model_pipeline)


#? Tải models
if not st.session_state.models_loaded:
    try:
        st.info("Đang tải models...")
        st.session_state.embeddings = load_embeddings()
        st.session_state.llm = load_llm()
        st.session_state.models_loaded = True
        st.success("Models đã sẵn sàng!")
        st.rerun()
    except Exception as e:
        st.error(f"Error loading models: {e}")
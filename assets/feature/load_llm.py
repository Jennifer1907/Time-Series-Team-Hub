import streamlit as st
import os
import torch
from dotenv import load_dotenv  # Load biến môi trường

load_dotenv()

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

# Lấy token HF từ biến môi trường (không cần cache)
@st.cache_resource
def get_hg_token():
<<<<<<< HEAD
    with open('token.txt', 'r') as f:
        hg_token = f.read()
        return hg_token


@st.cache_resource
def load_embeddings():
    """Tải mô hình embedding tiếng Việt"""
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
=======
    with open('token.txt', 'r') as f: #update token khi run vì token huggingface bị expire
        return f.read().strip()
>>>>>>> main

# Khởi tạo session state
if 'llm' not in st.session_state:
    st.session_state.llm = None

@st.cache_resource
def load_llm():
<<<<<<< HEAD
    MODEL_NAME= "lmsys/vicuna-7b-v1.5"
    # MODEL_NAME = "google/gemma-2b-it"

=======
    # Thay đổi model nếu google/gemma-2b không tồn tại hoặc private
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
>>>>>>> main

    token = get_hg_token()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_auth_token=token,
        torch_dtype=torch.float16,
        device_map="auto"  # để tự động chọn GPU/CPU
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=token)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        device_map="auto"
    )

    return HuggingFacePipeline(pipeline=model_pipeline)

<<<<<<< HEAD
=======
@st.cache_resource
def load_embeddings():
    # Tên model embedding nên kiểm tra có public không, đổi nếu cần
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder", 
                                 model_kwargs={"use_auth_token": get_hg_token()})
>>>>>>> main

if not st.session_state.get("models_loaded", False):
    st.info("Đang tải models...")
    st.session_state.embeddings = load_embeddings()
    st.session_state.llm = load_llm()
    st.session_state.models_loaded = True
    st.success("Models đã sẵn sàng!")
    st.experimental_rerun()

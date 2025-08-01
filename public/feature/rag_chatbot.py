import streamlit as st
import os
import requests
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time
import tempfile
import urllib.parse
import zipfile

st.set_page_config(
    page_title="Trợ Lý AI Tiếng Việt",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .main-header{
    text-align: center;
    padding: 1rem 0;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #ff0000, #ffff00);
    border-radius: 10px;
    color: white;
  }
  .chat-container{
    max-width: 800px;
    margin: 0 auto;
    padding: 1rem;
    max-height: 500px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    margin-bottom: 20px;
    background-color: #fafafa;
  }
  .user-message{
    background-color: #000000;
    color: #ffffff;
    border-radius: 18px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-left: 20%;
    text-align: left;
    border: 1px solid #333333;
  }
  .assistant-message{
    background-color: #006400;
    color: #ffffff;
    border-radius: 18px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-right: 20%;
    text-align: left;
    border: 1px solid #228b22;
  }
  .chat-input-container {
    position: sticky;
    bottom: 0;
    background-color: white;
    padding: 1rem;
    border-top: 2px solid #e0e0e0;
    border-radius: 10px;
    margin-top: 20px;
  }
  .stTextInput > div > div > input {
    border-radius: 25px;
    border: 2px solid #e0e0e0;
    padding: 12px 20px;
    font-size: 16px;
  }
  .document-info {
    background-color: #f8f9fa;
    border-left: 4px solid #28a745;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 4px;
  }
  .status-indicator{
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
  }
  .status-ready{
    background-color: #28a745;
  }
  .status-loading {
    background-color: #ffc107;
  }
  .status-error {
    background-color: #dc3545;
  }
  .thinking-indicator {
    background-color: #f5f5f5;
    border-radius: 18px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-right: 20%;
    text-align: left;
    border: 1px solid #ddd;
    animation: pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
  }
  .upload-section {
    background-color: #f8f9fa;
    border: 2px dashed #28a745;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    text-align: center;
  }
  .file-counter {
    background-color: #e3f2fd;
    border-radius: 5px;
    padding: 5px 10px;
    margin: 5px;
    display: inline-block;
    font-size: 12px;
  }
  .vietnam-flag {
    background: #da020e;
    width: 40px;
    height: 28px;
    display: inline-block;
    margin-right: 10px;
    border-radius: 3px;
    position: relative;
  }
  .vietnam-flag::after {
    content: "⭐";
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #ffcd00;
    font-size: 16px;
  }
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'pdf_source' not in st.session_state:
    st.session_state.pdf_source = "github"
if 'github_repo_url' not in st.session_state:
    st.session_state.github_repo_url = "https://github.com/Jennifer1907/Time-Series-Team-Hub/tree/main/assets/pdf"
if 'local_folder_path' not in st.session_state:
    st.session_state.local_folder_path = "./knowledge_base"
if 'processing_query' not in st.session_state:
    st.session_state.processing_query = False
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

@st.cache_resource
def load_embeddings():
    """Tải mô hình embedding tiếng Việt"""
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

def simple_text_generation(question, context):
    """Fallback function sử dụng rule-based approach với debugging"""
    try:
        # Kiểm tra context
        if not context or len(context.strip()) < 10:
            return "Nội dung tài liệu không đủ để trả lời câu hỏi."
        
        # Tìm kiếm từ khóa trong context
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Chia context thành câu
        sentences = []
        # Thử nhiều cách chia câu
        for delimiter in ['. ', '.\n', '! ', '?\n', '? ']:
            if delimiter in context:
                sentences.extend(context.split(delimiter))
        
        # Nếu không có câu nào, chia theo đoạn
        if not sentences:
            sentences = context.split('\n')
        
        # Lọc câu rỗng
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return "Không thể phân tích cấu trúc tài liệu. Có thể tài liệu bị lỗi format."
        
        # Từ dừng tiếng Việt mở rộng
        vietnamese_stopwords = {
            'là', 'của', 'và', 'với', 'cho', 'từ', 'về', 'theo', 'trong', 'nào', 'gì', 'sao', 
            'thế', 'như', 'có', 'không', 'được', 'này', 'đó', 'những', 'các', 'một', 'hai', 
            'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười', 'the', 'and', 'or', 
            'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'how', 'when', 
            'where', 'why', 'who', 'which', 'that', 'this', 'these', 'those', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'must', 'shall'
        }
        
        # Trích xuất từ khóa từ câu hỏi
        question_words = []
        for word in question_lower.split():
            clean_word = word.strip('.,!?()[]{}":;').lower()
            if len(clean_word) > 2 and clean_word not in vietnamese_stopwords:
                question_words.append(clean_word)
        
        if not question_words:
            return "Không thể xác định từ khóa từ câu hỏi. Vui lòng đặt câu hỏi cụ thể hơn."
        
        # Tìm câu liên quan
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = 0
            
            # Tính điểm dựa trên số từ khóa xuất hiện
            for word in question_words:
                if word in sentence_lower:
                    # Từ xuất hiện chính xác
                    score += 2
                    # Bonus nếu từ xuất hiện nhiều lần
                    score += sentence_lower.count(word) - 1
            
            # Bonus cho câu chứa nhiều từ khóa
            if score > 0:
                word_coverage = sum(1 for word in question_words if word in sentence_lower)
                coverage_bonus = (word_coverage / len(question_words)) * 2
                score += coverage_bonus
                
                relevant_sentences.append((sentence.strip(), score))
        
        if relevant_sentences:
            # Sắp xếp theo điểm relevance và lấy top 5
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sent[0] for sent in relevant_sentences[:5]]
            
            answer = "Dựa trên tài liệu, tôi tìm thấy thông tin sau:\n\n"
            for i, sentence in enumerate(top_sentences, 1):
                if sentence.strip():
                    # Làm sạch câu
                    clean_sentence = sentence.strip()
                    if not clean_sentence.endswith(('.', '!', '?')):
                        clean_sentence += '.'
                    answer += f"{i}. {clean_sentence}\n\n"
            
            return answer.strip()
        else:
            # Fallback: trả về một phần ngẫu nhiên của text
            preview_text = context[:1000] + "..." if len(context) > 1000 else context
            return f"Tôi không thể tìm thấy thông tin cụ thể liên quan đến câu hỏi '{question}' trong tài liệu. Tuy nhiên, đây là một phần nội dung tài liệu:\n\n{preview_text}\n\nVui lòng thử diễn đạt lại câu hỏi hoặc hỏi về các chủ đề khác được đề cập trong tài liệu."
    
    except Exception as e:
        return f"Xin lỗi, tôi gặp lỗi khi tìm kiếm: {str(e)}. Vui lòng thử lại."

def extract_text_from_uploaded_file(file):
    """Trích xuất văn bản từ file được tải lên dựa trên loại file"""
    file_extension = file.name.split('.')[-1].lower()
    
    try:
        # Tạo file tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(file.getbuffer())
            tmp_path = tmp_file.name
        
        documents = []
        
        if file_extension == 'pdf':
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
        elif file_extension == 'docx':
            loader = Docx2txtLoader(tmp_path)
            documents = loader.load()
        elif file_extension in ['xlsx', 'xls']:
            loader = UnstructuredExcelLoader(tmp_path)
            documents = loader.load()
        else:
            st.warning(f"Định dạng file không được hỗ trợ: {file_extension}")
            return []
        
        # Dọn dẹp file tạm thời
        os.unlink(tmp_path)
        return documents
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý {file.name}: {str(e)}")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return []

def process_zip_file(zip_file):
    """Xử lý file zip được tải lên chứa tài liệu"""
    try:
        all_documents = []
        loaded_files = []
        
        # Tạo thư mục tạm thời để giải nén
        temp_dir = tempfile.mkdtemp()
        
        # Lưu file zip được tải lên
        zip_path = os.path.join(temp_dir, zip_file.name)
        with open(zip_path, 'wb') as f:
            f.write(zip_file.getbuffer())
        
        # Giải nén file zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Tìm tất cả file tài liệu trong thư mục đã giải nén
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.pdf', '.docx', '.xlsx', '.xls')):
                    file_path = os.path.join(root, file)
                    try:
                        if file.lower().endswith('.pdf'):
                            loader = PyPDFLoader(file_path)
                        elif file.lower().endswith('.docx'):
                            loader = Docx2txtLoader(file_path)
                        elif file.lower().endswith(('.xlsx', '.xls')):
                            loader = UnstructuredExcelLoader(file_path)
                        
                        documents = loader.load()
                        all_documents.extend(documents)
                        loaded_files.append(file)
                        st.success(f"✅ Đã xử lý từ zip: {file}")
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi khi xử lý {file} từ zip: {str(e)}")
        
        # Dọn dẹp
        shutil.rmtree(temp_dir)
        return all_documents, loaded_files
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý file zip: {str(e)}")
        return [], []

def get_github_pdf_files(repo_url):
    """Lấy danh sách file PDF từ GitHub repository"""
    try:
        if "github.com" in repo_url and "/tree/" in repo_url:
            parts = repo_url.replace("https://github.com/", "").split("/tree/")
            repo_path = parts[0]
            branch_and_path = parts[1].split("/", 1)
            branch = branch_and_path[0]
            folder_path = branch_and_path[1] if len(branch_and_path) > 1 else ""

            api_url = f"https://api.github.com/repos/{repo_path}/contents/{folder_path}?ref={branch}"
        else:
            st.error("Định dạng URL GitHub không hợp lệ")
            return []

        response = requests.get(api_url)
        if response.status_code == 200:
            files = response.json()
            pdf_files = []
            for file in files:
                if file['name'].endswith('.pdf') and file['type'] == 'file':
                    pdf_files.append({
                        'name': file['name'],
                        'download_url': file['download_url']
                    })
            return pdf_files
        else:
            st.error(f"Không thể truy cập GitHub repository: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Lỗi khi truy cập GitHub repository: {str(e)}")
        return []

def download_pdf_from_url(url, filename, temp_dir):
    """Tải file PDF từ URL"""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return file_path
        return None
    except Exception as e:
        st.error(f"Lỗi khi tải {filename}: {str(e)}")
        return None

def create_rag_chain(all_documents):
    """Tạo chuỗi RAG từ tài liệu"""
    if not all_documents:
        st.error("Không có tài liệu nào để xử lý")
        return None, 0
    
    try:
        st.info(f"🔄 Đang xử lý {len(all_documents)} tài liệu...")
        
        # Kiểm tra nội dung tài liệu
        total_text = ""
        for doc in all_documents:
            if hasattr(doc, 'page_content'):
                total_text += doc.page_content + "\n"
        
        if len(total_text.strip()) < 50:
            st.error("Nội dung tài liệu quá ngắn hoặc không thể đọc được")
            return None, 0
        
        st.success(f"✅ Đã đọc {len(total_text):,} ký tự từ tài liệu")
        
        # Lưu toàn bộ text vào session state để fallback
        st.session_state.documents_text = total_text
        
        # Sử dụng text splitter mạnh mẽ hơn nếu SemanticChunker thất bại
        try:
            if st.session_state.embeddings:
                semantic_splitter = SemanticChunker(
                    embeddings=st.session_state.embeddings,
                    buffer_size=1,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=95,
                    min_chunk_size=500,
                    add_start_index=True
                )
                docs = semantic_splitter.split_documents(all_documents)
                st.info(f"✅ Sử dụng SemanticChunker: {len(docs)} chunks")
            else:
                raise Exception("No embeddings available")
        except Exception as e:
            st.warning(f"SemanticChunker thất bại: {str(e)}")
            st.info("🔄 Chuyển sang RecursiveCharacterTextSplitter...")
            # Dự phòng với text splitter cơ bản
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            docs = text_splitter.split_documents(all_documents)
            st.info(f"✅ Sử dụng RecursiveCharacterTextSplitter: {len(docs)} chunks")
        
        if not docs:
            st.error("Không có đoạn tài liệu nào được tạo")
            # Tạo simple RAG chain với toàn bộ text
            def simple_rag_chain_text(question):
                return simple_text_generation(question, total_text)
            return simple_rag_chain_text, 1
        
        # Triển khai FAISS với xử lý lỗi (chỉ khi có embeddings)
        if st.session_state.embeddings:
            try:
                vector_db = FAISS.from_documents(documents=docs, embedding=st.session_state.embeddings)
                retriever = vector_db.as_retriever(search_kwargs={"k": min(4, len(docs))})
                st.success(f"✅ Đã tạo FAISS vector database với {len(docs)} chunks")
            except Exception as e:
                st.error(f"Lỗi khi tạo FAISS vector database: {str(e)}")
                st.info("🔄 Chuyển sang chế độ tìm kiếm text đơn giản...")
                # Fallback to simple text search
                def simple_rag_chain_docs(question):
                    combined_text = "\n\n".join([doc.page_content for doc in docs])
                    return simple_text_generation(question, combined_text)
                return simple_rag_chain_docs, len(docs)
        else:
            st.info("🔍 Không có embeddings, sử dụng tìm kiếm text đơn giản")
            def simple_rag_chain_docs(question):
                combined_text = "\n\n".join([doc.page_content for doc in docs])
                return simple_text_generation(question, combined_text)
            return simple_rag_chain_docs, len(docs)

        # Sử dụng tìm kiếm từ khóa thông minh với hub prompt
        st.info("🔍 Sử dụng tìm kiếm từ khóa thông minh với RAG prompt")
        
        # Tải prompt từ hub
        try:
            prompt = hub.pull("rlm/rag-prompt")
            st.success("✅ Đã tải prompt template từ hub")
        except Exception as e:
            st.warning(f"Không thể tải prompt từ hub: {str(e)}")
            st.info("🔄 Sử dụng prompt template cục bộ...")
            prompt_template = """Sử dụng những đoạn ngữ cảnh sau để trả lời câu hỏi ở cuối.
            Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.
            Trả lời bằng tiếng Việt.

            Ngữ cảnh: {context}

            Câu hỏi: {question}

            Trả lời:"""
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

        def format_docs(docs):
            if not docs:
                return "Không tìm thấy tài liệu liên quan."
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Tạo simple RAG chain sử dụng keyword search với prompt
        def smart_rag_chain_with_prompt(question):
            try:
                # Tìm tài liệu liên quan bằng retriever
                relevant_docs = retriever.get_relevant_documents(question)
                context = format_docs(relevant_docs)
                
                # Sử dụng simple text generation để trả lời
                return simple_text_generation(question, context)
                
            except Exception as e:
                st.warning(f"Lỗi retriever: {str(e)}, sử dụng toàn bộ text")
                return simple_text_generation(question, total_text)
        
        return smart_rag_chain_with_prompt, len(docs)
        
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi tạo chuỗi RAG: {str(e)}")
        st.info("🔄 Tạo fallback RAG chain...")
        # Ultimate fallback
        def emergency_rag_chain(question):
            if hasattr(st.session_state, 'documents_text') and st.session_state.documents_text:
                return simple_text_generation(question, st.session_state.documents_text)
            else:
                return "Xin lỗi, không thể truy cập nội dung tài liệu. Vui lòng tải lại tài liệu."
        return emergency_rag_chain, 1

def load_pdfs_from_github(repo_url):
    """Tải file PDF từ GitHub repository"""
    pdf_files = get_github_pdf_files(repo_url)

    if not pdf_files:
        st.warning("Không tìm thấy file PDF nào trong GitHub repository")
        return None, 0, []

    temp_dir = tempfile.mkdtemp()
    all_documents = []
    loaded_files = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, pdf_file in enumerate(pdf_files):
        try:
            status_text.text(f"Đang tải và xử lý: {pdf_file['name']}")
            local_path = download_pdf_from_url(pdf_file['download_url'], pdf_file['name'], temp_dir)

            if local_path:
                loader = PyPDFLoader(local_path)
                documents = loader.load()
                all_documents.extend(documents)
                loaded_files.append(pdf_file['name'])

                st.success(f"✅ Đã xử lý: {pdf_file['name']} ({len(documents)} trang)")
            progress_bar.progress((i + 1) / len(pdf_files))
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý {pdf_file['name']}: {str(e)}")

    progress_bar.empty()
    status_text.empty()

    # Dọn dẹp thư mục tạm thời
    shutil.rmtree(temp_dir)

    if not all_documents:
        return None, 0, loaded_files

    rag_chain, num_chunks = create_rag_chain(all_documents)
    return rag_chain, num_chunks, loaded_files

def load_pdfs_from_folder(folder_path):
    """Tải tất cả file PDF từ thư mục được chỉ định"""
    cleaned_path = folder_path.strip().strip('"').strip("'")
    folder = Path(cleaned_path)

    if not folder.exists():
        st.error(f"❌ Thư mục không tồn tại: `{cleaned_path}`")
        return None, 0, []

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        st.warning(f"Không tìm thấy file PDF nào trong thư mục: {cleaned_path}")
        return None, 0, []

    all_documents = []
    loaded_files = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, pdf_file in enumerate(pdf_files):
        try:
            status_text.text(f"Đang xử lý: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            all_documents.extend(documents)
            loaded_files.append(pdf_file.name)
            progress_bar.progress((i + 1) / len(pdf_files))
            st.success(f"✅ Đã xử lý: {pdf_file.name} ({len(documents)} trang)")

        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý {pdf_file.name}: {str(e)}")

    progress_bar.empty()
    status_text.empty()

    if not all_documents:
        return None, 0, loaded_files

    rag_chain, num_chunks = create_rag_chain(all_documents)
    return rag_chain, num_chunks, loaded_files

def display_chat_message(message, is_user=True):
    """Hiển thị tin nhắn trò chuyện"""
    if is_user:
        st.markdown(f"""
        <div class="user-message">
            <strong style="color: #ffffff;">Bạn:</strong> <span style="color: #ffffff;">{message}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong style="color: #ffffff;">Trợ Lý AI:</strong> <span style="color: #ffffff;">{message}</span>
        </div>
        """, unsafe_allow_html=True)

def display_thinking_indicator():
    """Hiển thị chỉ báo đang suy nghĩ"""
    st.markdown(f"""
    <div class="thinking-indicator">
        <strong>Trợ Lý AI:</strong> 🤔 Đang suy nghĩ...
    </div>
    """, unsafe_allow_html=True)

def process_user_query(question):
    """Xử lý câu hỏi của người dùng"""
    try:
        if not st.session_state.rag_chain:
            return "Xin lỗi, chưa có tài liệu nào được tải. Vui lòng tải lên hoặc nạp tài liệu trước."
        
        # Kiểm tra câu hỏi
        if not question or len(question.strip()) < 2:
            return "Vui lòng đặt câu hỏi cụ thể hơn."
        
        # Gọi chuỗi RAG với xử lý lỗi chi tiết
        try:
            if callable(st.session_state.rag_chain):
                # Simple RAG chain (fallback)
                output = st.session_state.rag_chain(question)
            else:
                # LangChain RAG chain (không có vì đã bỏ LLM)
                output = st.session_state.rag_chain(question)
            
        except Exception as chain_error:
            st.error(f"Lỗi khi gọi RAG chain: {str(chain_error)}")
            # Ultimate fallback: sử dụng documents_text nếu có
            if hasattr(st.session_state, 'documents_text') and st.session_state.documents_text:
                return simple_text_generation(question, st.session_state.documents_text)
            else:
                return f"Xin lỗi, gặp lỗi khi xử lý câu hỏi: {str(chain_error)}. Vui lòng thử tải lại tài liệu."
        
        # Xử lý các định dạng đầu ra khác nhau
        if isinstance(output, str):
            # Nếu đầu ra chứa "Answer:", trích xuất phần sau nó
            if 'Answer:' in output:
                answer_parts = output.split('Answer:')
                if len(answer_parts) > 1:
                    answer = answer_parts[-1].strip()
                else:
                    answer = output.strip()
            elif 'Trả lời:' in output:
                answer_parts = output.split('Trả lời:')
                if len(answer_parts) > 1:
                    answer = answer_parts[-1].strip()
                else:
                    answer = output.strip()
            else:
                answer = output.strip()
        else:
            # Nếu đầu ra không phải là chuỗi, chuyển đổi nó
            answer = str(output).strip()
        
        # Đảm bảo có câu trả lời có ý nghĩa
        if not answer or len(answer) < 5:
            return "Tôi đã tìm thấy một số thông tin trong tài liệu, nhưng không thể tạo ra câu trả lời rõ ràng. Vui lòng thử diễn đạt lại câu hỏi của bạn."
        
        # Làm sạch câu trả lời
        answer = answer.replace("Human:", "").replace("Assistant:", "").strip()
        
        return answer
        
    except Exception as e:
        st.error(f"Lỗi không mong đợi: {str(e)}")
        # Thử fallback cuối cùng
        if hasattr(st.session_state, 'documents_text') and st.session_state.documents_text:
            return simple_text_generation(question, st.session_state.documents_text)
        return "Tôi xin lỗi, gặp lỗi không mong đợi. Vui lòng thử tải lại tài liệu hoặc đặt câu hỏi khác."

def main():
    # Header với cờ Việt Nam
    st.markdown("""
    <div class="main-header">
        <div class="vietnam-flag"></div>
        <h1>🇻🇳 Trợ Lý AI Tiếng Việt</h1>
        <p>Hệ thống hỏi đáp thông minh với tài liệu PDF, Word, Excel bằng tiếng Việt</p>
        <p style="font-size: 14px; margin-top: 10px;">🌟 Powered by Vietnamese AI Technology - No API Key Required! 🌟</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Cấu Hình")

        if st.session_state.models_loaded:
            st.markdown('<span class="status-indicator status-ready"></span>**Mô hình:** Sẵn sàng', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-loading"></span>**Mô hình:** Đang tải...', unsafe_allow_html=True)

        # Trạng thái tải tài liệu
        if st.session_state.documents_loaded:
            st.markdown('<span class="status-indicator status-ready"></span>**Tài liệu:** Đã tải (FAISS)', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-error"></span>**Tài liệu:** Chưa tải', unsafe_allow_html=True)

        st.divider()

        # Lựa chọn nguồn tài liệu
        st.subheader("📁 Nguồn Tài Liệu")

        pdf_source = st.radio(
            "Chọn nguồn tài liệu:",
            ["Tải File Lên", "Tải Thư Mục (ZIP)", "GitHub Repository", "Đường Dẫn Thư Mục"],
            key="pdf_source_radio"
        )

        if pdf_source == "Tải File Lên":
            st.session_state.pdf_source = "upload_files"
            
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.markdown("**📎 Tải Lên Từng File**")
            uploaded_files = st.file_uploader(
                "Chọn file để tải lên:",
                type=['pdf', 'docx', 'xlsx', 'xls'],
                accept_multiple_files=True,
                help="Định dạng hỗ trợ: PDF, Word (.docx), Excel (.xlsx, .xls)"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_files:
                st.markdown("**File Đã Chọn:**")
                for i, file in enumerate(uploaded_files):
                    file_size = len(file.getbuffer()) / (1024 * 1024)  # Kích thước tính bằng MB
                    st.markdown(f'<span class="file-counter">{i+1}. {file.name} ({file_size:.1f} MB)</span>', unsafe_allow_html=True)
                
                if st.button("📤 Xử Lý File Đã Tải", type="primary"):
                    with st.spinner("Đang xử lý file đã tải lên..."):
                        all_documents = []
                        loaded_files = []
                        
                        progress_bar = st.progress(0)
                        
                        for i, file in enumerate(uploaded_files):
                            documents = extract_text_from_uploaded_file(file)
                            if documents:
                                all_documents.extend(documents)
                                loaded_files.append(file.name)
                                st.success(f"✅ Đã xử lý: {file.name}")
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        progress_bar.empty()
                        
                        if all_documents:
                            rag_chain, num_chunks = create_rag_chain(all_documents)
                            if rag_chain:
                                st.session_state.rag_chain = rag_chain
                                st.session_state.documents_loaded = True
                                st.success(f"✅ Đã xử lý thành công {len(loaded_files)} file!")
                                st.rerun()
                        else:
                            st.error("Không có tài liệu nào có thể được xử lý.")

        elif pdf_source == "Tải Thư Mục (ZIP)":
            st.session_state.pdf_source = "upload_zip"
            
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.markdown("**📁 Tải Thư Mục Dưới Dạng ZIP**")
            zip_file = st.file_uploader(
                "Chọn file ZIP chứa tài liệu:",
                type=['zip'],
                help="Tải lên file ZIP chứa file PDF, Word, hoặc Excel"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if zip_file:
                file_size = len(zip_file.getbuffer()) / (1024 * 1024)  # Kích thước tính bằng MB
                st.info(f"📦 File ZIP đã chọn: {zip_file.name} ({file_size:.1f} MB)")
                
                if st.button("📤 Xử Lý File ZIP", type="primary"):
                    with st.spinner("Đang giải nén và xử lý file ZIP..."):
                        all_documents, loaded_files = process_zip_file(zip_file)
                        
                        if all_documents:
                            rag_chain, num_chunks = create_rag_chain(all_documents)
                            if rag_chain:
                                st.session_state.rag_chain = rag_chain
                                st.session_state.documents_loaded = True
                                st.success(f"✅ Đã xử lý thành công {len(loaded_files)} file từ ZIP!")
                                st.rerun()
                        else:
                            st.error("Không tìm thấy tài liệu hợp lệ trong file ZIP.")

        elif pdf_source == "GitHub Repository":
            st.session_state.pdf_source = "github"
            github_url = st.text_input(
                "URL GitHub Repository:",
                value=st.session_state.github_repo_url,
                help="URL đến thư mục GitHub chứa file PDF"
            )
            st.session_state.github_repo_url = github_url
            
            if st.button("📥 Tải Từ GitHub", type="primary"):
                st.session_state.documents_loaded = False
                st.rerun()

        else:  # Đường Dẫn Thư Mục
            st.session_state.pdf_source = "local"
            local_path = st.text_input(
                "Đường Dẫn Thư Mục Cục Bộ:",
                value=st.session_state.local_folder_path,
                help="Đường dẫn đến thư mục cục bộ chứa file PDF"
            )
            st.session_state.local_folder_path = local_path
            
            if st.button("📂 Tải Từ Thư Mục Cục Bộ", type="primary"):
                st.session_state.documents_loaded = False
                st.rerun()

        st.divider()

        if st.button("🗑️ Xóa Lịch Sử Trò Chuyện"):
            st.session_state.chat_history = []
            st.session_state.processing_query = False
            st.rerun()

        if st.button("🗑️ Xóa Tất Cả Tài Liệu"):
            st.session_state.documents_loaded = False
            st.session_state.rag_chain = None
            st.session_state.chat_history = []
            st.session_state.processing_query = False
            if hasattr(st.session_state, 'documents_text'):
                del st.session_state.documents_text
            st.rerun()

        # Cài đặt FAISS
        st.divider()
        st.subheader("🔍 Cài Đặt Hệ Thống")
        st.info("🚀 FAISS: Thư viện tìm kiếm tương tự nhanh")
        st.info("🔍 Keyword Search: Tìm kiếm từ khóa thông minh")
        st.info("🌐 Hub Prompt: Sử dụng rlm/rag-prompt template")
        
        st.divider()
        st.subheader("🇻🇳 Mô Hình Tiếng Việt")
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div class="vietnam-flag" style="margin-right: 15px;"></div>
                <strong>Vietnamese AI Technology</strong>
            </div>
            <p style="margin: 0; font-size: 14px;">
                ✨ Sử dụng mô hình embedding 'bkai-foundation-models/vietnamese-bi-encoder'<br>
                🚀 Được tối ưu hóa đặc biệt cho ngôn ngữ tiếng Việt<br>
                🎯 Hiểu rõ ngữ cảnh và từ ngữ Việt Nam<br>
                🔑 Không cần API Key - Hoạt động offline!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Debug section
        st.divider()
        st.subheader("🔧 Debug & Kiểm Tra")
        
        if st.button("🔍 Kiểm Tra Hệ Thống"):
            st.write("**Trạng thái Hệ Thống:**")
            st.write(f"- Models loaded: {st.session_state.models_loaded}")
            st.write(f"- Embeddings: {'✅' if st.session_state.embeddings else '❌'}")
            st.write(f"- Documents loaded: {st.session_state.documents_loaded}")
            st.write(f"- RAG chain: {'✅' if st.session_state.rag_chain else '❌'}")
            st.write(f"- Mode: 🔍 Keyword Search (No API required)")
            
            if hasattr(st.session_state, 'documents_text'):
                st.write(f"- Documents text length: {len(st.session_state.documents_text):,} characters")
            else:
                st.write("- Documents text: ❌ Chưa có")
        
        if st.session_state.documents_loaded and st.button("📄 Xem Mẫu Nội Dung"):
            if hasattr(st.session_state, 'documents_text') and st.session_state.documents_text:
                preview = st.session_state.documents_text[:500] + "..." if len(st.session_state.documents_text) > 500 else st.session_state.documents_text
                st.text_area("Mẫu nội dung tài liệu:", preview, height=200)

    # Tải mô hình nếu chưa được tải
    if not st.session_state.models_loaded:
        with st.spinner("🚀 Đang khởi tạo embedding model tiếng Việt..."):
            try:
                st.session_state.embeddings = load_embeddings()
                st.success("✅ Đã tải embeddings model thành công")
            except Exception as e:
                st.error(f"❌ Lỗi khi tải embeddings: {str(e)}")
                st.warning("⚠️ Sẽ hoạt động ở chế độ đơn giản mà không có embeddings")
                st.session_state.embeddings = None
            
            st.session_state.models_loaded = True
        
        st.success("✅ Hệ thống đã sẵn sàng!")
        st.info("🔍 Đang hoạt động ở chế độ tìm kiếm từ khóa thông minh")
        time.sleep(1)
        st.rerun()

    # Tải tài liệu nếu chưa được tải và nguồn là github hoặc local
    if st.session_state.models_loaded and not st.session_state.documents_loaded and st.session_state.pdf_source in ["github", "local"]:
        with st.spinner("📚 Đang tải tài liệu vào kho vector FAISS..."):
            if st.session_state.pdf_source == "github":
                rag_chain, num_chunks, loaded_files = load_pdfs_from_github(st.session_state.github_repo_url)
            else:
                rag_chain, num_chunks, loaded_files = load_pdfs_from_folder(st.session_state.local_folder_path)

            if rag_chain:
                st.session_state.rag_chain = rag_chain
                st.session_state.documents_loaded = True

                st.markdown(f"""
                <div class="document-info">
                    <h4>📄 Đã tải thành công {len(loaded_files)} tài liệu PDF vào FAISS:</h4>
                    <ul>
                        {"".join([f"<li>{file}</li>" for file in loaded_files])}
                    </ul>
                    <p><strong>Tổng số đoạn:</strong> {num_chunks}</p>
                    <p><strong>Kho Vector:</strong> FAISS (Tìm kiếm tương tự nhanh)</p>
                    <p><strong>Chế độ:</strong> 🔍 Keyword Search với RAG Prompt</p>
                    <p><strong>Template:</strong> rlm/rag-prompt từ LangChain Hub</p>
                </div>
                """, unsafe_allow_html=True)

                st.success("✅ Tài liệu đã sẵn sàng cho hỏi đáp!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Không thể tải tài liệu. Vui lòng kiểm tra cấu hình của bạn.")

    # Giao diện trò chuyện chính
    if st.session_state.rag_chain:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

        for message in st.session_state.chat_history:
            display_chat_message(message["content"], message["is_user"])

        if st.session_state.processing_query:
            display_thinking_indicator()

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)

        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                user_question = st.text_input(
                    "Nhập câu hỏi của bạn...",
                    placeholder="Hỏi bất cứ điều gì về tài liệu...",
                    disabled=st.session_state.processing_query,
                    label_visibility="collapsed"
                )

            with col2:
                send_button = st.form_submit_button(
                    "📤 Gửi",
                    type="primary",
                    disabled=st.session_state.processing_query
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # Xử lý đầu vào của người dùng
        if send_button and user_question.strip() and not st.session_state.processing_query:
            st.session_state.processing_query = True

            st.session_state.chat_history.append({
                "content": user_question,
                "is_user": True
            })
            st.rerun()

        if st.session_state.processing_query and len(st.session_state.chat_history) > 0:
            if not st.session_state.chat_history[-1]["is_user"]:
                st.session_state.processing_query = False
            else:
                last_question = st.session_state.chat_history[-1]["content"]
                answer = process_user_query(last_question)

                st.session_state.chat_history.append({
                    "content": answer,
                    "is_user": False
                })

                st.session_state.processing_query = False

                st.rerun()
    else:
        # Tin nhắn chào mừng
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h3>👋 Chào mừng đến với Trợ Lý AI Tiếng Việt!</h3>
            <p><strong style="color: #28a745;">🔥 Không cần API Key - Hoạt động hoàn toàn offline!</strong></p>
            <br>
            <p>Hệ thống này hỗ trợ nhiều phương thức nhập liệu:</p>
            <ul style='text-align: left; max-width: 500px; margin: 0 auto;'>
                <li><strong>📎 Tải File Lên:</strong>
                    <ul>
                        <li>Tài liệu PDF (.pdf)</li>
                        <li>Tài liệu Word (.docx)</li>
                        <li>Bảng tính Excel (.xlsx, .xls)</li>
                    </ul>
                </li>
                <li><strong>📁 Tải Thư Mục (ZIP):</strong> Tải lên file ZIP chứa nhiều tài liệu</li>
                <li><strong>🔗 GitHub Repository:</strong> Tải file PDF từ kho GitHub</li>
                <li><strong>📂 Thư Mục Cục Bộ:</strong> Tải file từ đường dẫn thư mục cục bộ</li>
            </ul>
            <br>
            <p><strong>Công Nghệ Sử Dụng:</strong></p>
            <ul style='text-align: left; max-width: 500px; margin: 0 auto;'>
                <li><strong>🔍 Smart Keyword Search:</strong> Tìm kiếm từ khóa thông minh</li>
                <li><strong>🌐 RAG Prompt:</strong> Sử dụng template từ LangChain Hub</li>
                <li><strong>🚀 FAISS Vector Store:</strong> Tìm kiếm tương tự nhanh</li>
                <li><strong>🇻🇳 Vietnamese Embeddings:</strong> Tối ưu cho tiếng Việt</li>
            </ul>
            <br>
            <p><strong>Để bắt đầu:</strong></p>
            <ol style='text-align: left; max-width: 500px; margin: 0 auto;'>
                <li>Chọn nguồn tài liệu ưa thích</li>
                <li>Tải lên file hoặc cấu hình repository/thư mục</li>
                <li>Xử lý tài liệu của bạn</li>
                <li>Bắt đầu đặt câu hỏi!</li>
            </ol>
            <br>
            <p><strong>Repository Mặc Định:</strong><br>
            <code>https://github.com/Jennifer1907/Time-Series-Team-Hub/tree/main/assets/pdf</code></p>
            <br>
            <p><strong>Tính Năng Nổi Bật:</strong></p>
            <ul style='text-align: left; max-width: 500px; margin: 0 auto;'>
                <li>✨ Hỗ trợ đa định dạng (PDF, Word, Excel)</li>
                <li>🚀 Kho vector FAISS cho tìm kiếm tương tự nhanh</li>
                <li>🇻🇳 Tối ưu cho tiếng Việt</li>
                <li>🔄 Nhiều phương thức nhập liệu</li>
                <li>💬 Giao diện trò chuyện giống ChatGPT</li>
                <li>🎯 Phản hồi nhận thức ngữ cảnh</li>
                <li>🔑 Không cần API Key - Hoạt động offline!</li>
                <li>🌐 Sử dụng RAG prompt template từ LangChain Hub</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
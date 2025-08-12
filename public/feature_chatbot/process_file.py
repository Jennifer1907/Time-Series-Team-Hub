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
from langchain_community.vectorstores import FAISS
import tempfile
import zipfile



def simple_text_retrieval(question, context):
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

    # Tạo file tạm thời
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
        tmp_file.write(file.getbuffer())
        tmp_path = tmp_file.name

    try:

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
                        #? variable is only defined inside the if/elif conditions,
                        #? but it's used outside of them. If none of the conditions match, loader would be undefined when you try to call
                        else:
                            st.warning(f"Định dạng file không được hỗ trợ trong zip: {file}")
                            continue

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
---
title: "Module 1 - Tuần 4 - Trợ lý Gạo AI đã xuất hiện"
date: 2025-06-28T13:03:07+07:00
description: Trợ lý AI tiếng Việt hỗ trợ hỏi đáp từ tài liệu PDF bằng công nghệ RAG kết hợp mô hình Vicuna-7B, được xây dựng bằng Streamlit và LangChain.
image: images/RAGChatbot.png
caption:
categories:
  - minutes
tags:
  
draft: false
---

## 🤖 Trợ Lý Gạo AI Tiếng Việt — PDF RAG Assistant

Chào mừng bạn đến với chatbot Gạo AI thông minh của nhóm, được huấn luyện để **trả lời câu hỏi từ tài liệu PDF** bằng tiếng Việt.

## 🧪 Trải nghiệm Gạo AI tại đây: **[Streamlit](https://gaochatbot.streamlit.app/)** 

Vì streamlit deploy cloud không hỗ trợ một số thư viện nên dẫn đến một số hạn chế nếu bạn trải nghiệm tại web nên để có thể sử dụng tối đa chức năng của Gạo AI bạn có thể sử dụng code sau: [Github](https://github.com/Jennifer1907/Time-Series-Team-Hub/tree/main/assets/feature_chatbot)
- **💻 Với Gạo AI đọc và tóm tắt tài liệu**: Sử dụng file rag_chatbot.py
- **🔗 Với Gạo AI đọc tài liệu và thiết kế câu hỏi ôn tập**: Sử dụng file rag_chatbot2.py

👉 **Bạn có thể hỏi:**

- Với nội dung cho cá nhân: Bạn có thể tải lên một văn bản hoặc đường dẫn tiếng việt và đặt câu hỏi xung quanh tài liệu đó, Trợ lý AI sẽ giúp bạn đưa ra thông tin liên quan

  - 📚 **Sinh viên, học viên**: Tải tài liệu học hoặc chọn link kiến thức tổng hợp từ lớp AIO để ôn tập.
  - 🧑‍💼 **Người đi làm**: Upload văn bản nội bộ như hợp đồng, báo cáo,... và đặt câu hỏi để trích xuất thông tin nhanh.
  - 👨‍🏫 **Giảng viên**: Chuẩn bị nội dung tài liệu và để chatbot giúp trả lời cho người học dựa trên nội dung đó.

- Với nội dung lớp AIO từ Tuần 1 đến giờ: Bạn chọn phần Git Respository, tại đó nhóm có đặt link mặc định đến blog kiến thức tổng hợp của lớp và bạn có thể đặt câu hỏi để Trợ lý AI có thể giúp bạn ôn lại kiến thức liên quan AIO

---

### 🧠 Cách thức hoạt động

Chatbot được xây dựng trên nền tảng:

- **💻 Streamlit**: Tạo giao diện web đơn giản, dễ sử dụng.
- **🔗 LangChain**: Quản lý pipeline xử lý RAG.
- **🤗 HuggingFace + FAISS**: Truy hồi và xử lý dữ liệu ngữ nghĩa.
- **🧠 RAG (Retrieval-Augmented Generation)**: Kết hợp truy hồi tài liệu và sinh ngôn ngữ từ LLM.

---

## 🔍 Bạn có thể làm gì với chatbot này?

### ✅ Các chế độ hỗ trợ:

- 📄 **Tải lên tài liệu cá nhân**: Tài liệu dạng `.pdf`, `.docx`, `.xlsx`, `.zip` chứa nhiều file.
- 🔗 **Chọn tài liệu AIO từ GitHub**: Sử dụng link mặc định đến blog kiến thức tổng hợp của lớp AIO để hỏi bài.
- 💬 **Đặt câu hỏi bằng tiếng Việt** và nhận lại câu trả lời chính xác, có ngữ cảnh.
- 📄 **Sinh câu hỏi trắc nghiệm** và nhận lại câu hỏi trắc nghiệm để ôn tập kiến thức liên quan tài liệu.

---

## ♻️ Ưu điểm của Trợ lý AI:

### 🔍 Xử lý nhiều tài liệu, nhiều định dạng

- ✅ **Hỗ trợ .zip**: Cho phép nén nhiều tài liệu lại để tải lên cùng lúc.
- ✅ **Tự động nhận dạng**:
  - `.pdf` → `PyPDFLoader`
  - `.docx` → `Docx2txtLoader`
  - `.xlsx`, `.xls` → `UnstructuredExcelLoader`

### 🌐 Lấy tài liệu từ GitHub

- Chỉ cần dán link đến repository GitHub chứa các file `.pdf`
- Hệ thống gọi GitHub API, liệt kê và xử lý từng file trực tiếp.

### 🧠 Fallback thông minh khi không có LLM

Khi không thể gọi mô hình sinh ngôn ngữ (do lỗi hoặc cần debug), chatbot có thể dùng phương pháp `simple_text_generation()` để:

- Trích xuất từ khóa từ câu hỏi
- Tìm các câu trong tài liệu chứa nhiều từ khóa
- Trả về top 5 câu liên quan nhất
- Nếu không có kết quả, sẽ trả về đoạn preview và gợi ý hỏi lại

### 🔤 Embedding tối ưu cho tiếng Việt

Hệ thống sử dụng:

```python
HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
```

---

### 🛠️ Công nghệ sử dụng

| Thành phần             | Công cụ                                                                                               |
| ---------------------- | ------------------------------------ |
| 🧪 Trải nghiệm Gạo AI  | [Streamlit](https://gaochatbot.streamlit.app/)             |
| Code                   | [Google Colab](https://colab.research.google.com/drive/1RIqEgrFcSYTO6rlUj1jLoUuJFrtpZy4X?usp=sharing) |
| NLP model              | [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5)                                              |
| Embedding tiếng Việt   | `bkai-foundation-models/vietnamese-bi-encoder`                                                        |
| Xử lý PDF              | LangChain `PyPDFLoader`                                                                               |
| Semantic Split         | LangChain `SemanticChunker`                                                                           |
| Truy xuất văn bản      | ChromaDB                                                                                              |
| Truy vấn ngữ cảnh      | RAG pipeline                                                                                          |

---

📂 _Tài liệu đi kèm:_
{{< pdf src="/Time-Series-Team-Hub/pdf/M1W4D1_Project_Streamlit_RAG_Chatbot.pdf" title="Week_4_Streamlit_RAGChatbot" height="700px" >}}
{{< pdf src="/Time-Series-Team-Hub/pdf/M1W4D1_Project_Chatbot_Slide.pdf" title="Week_4_ChatbotSlide" height="700px" >}}


🧠 _Repository managed by [Time Series Team Hub](https://github.com/Jennifer1907/Time-Series-Team-Hub)_

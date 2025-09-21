---
title: "Module 2 - Tuần 4 - Dũng sĩ diệt SPAM đã trở lại và lợi hại hơn xưa"  
date: 2025-08-02T10:00:00+07:00  
description: Hệ thống phân loại tin nhắn spam/ham nâng cao với khả năng giải thích, được xây dựng bằng Streamlit và các kỹ thuật học máy/học sâu hiện đại.  
image: images/spam_detector.jpg
caption:  
categories:  
  - minutes  
tags:  
   
draft: false  

---

## 🤖 Module 2 - Tuần 4 - 🛡️ Dũng Sĩ Diệt SPAM đã trở lại

Chào mừng bạn đến với **Dũng Sĩ Diệt SPAM** của Time-Series Team! Đây là một công cụ tiên tiến giúp **phân loại tin nhắn thành spam hoặc ham** với độ chính xác cao và cung cấp **giải thích chi tiết** về cách hệ thống đưa ra quyết định. Hệ thống này không chỉ lọc tin nhắn rác mà còn mang lại sự minh bạch trong quá trình xử lý dữ liệu.

Hệ thống Dũng Sĩ Diệt SPAM sử dụng 3 cấp độ phân loại từ cơ bản đến nâng cao:

🎯 Phân loại tin nhắn cơ bản (HAM vs SPAM)
🔍 Phân loại với giải thích (Explainable AI)
🏷️ Phân loại phụ chi tiết (Spam Subcategorization)

---

## 🧪 Trải nghiệm Dũng sĩ diệt spam tại đây: 
**Dùng thử ngay**: **[Streamlit](https://superherospamai.streamlit.app/)** 

Vì một số hạn chế khi deploy lên Streamlit Cloud nên có thể sẽ xảy ra lỗi nếu cùng nhiều người vào một lúc. Nếu trường hợp xảy ra thì cho nhóm xin lỗi và bạn có thể clone code của nhóm từ Github về máy để chạy local nhé. 

Bên cạnh đó app đang thực hiện rất tốt phần Tiếng Việt, về phần Tiếng Anh nhóm có áp dụng Data Augmentation trong quá trình train model nên có thể hay crash trên Streamlit vì vậy bạn clone Github về máy chạy luôn nhé!

### 🧪 File Source Code: 
[Code_Streamlit] (https://github.com/Jennifer1907/Time-Series-Team-Hub/tree/main/assets/feature_spamdetector)  
[Google_Colab] (https://colab.research.google.com/drive/1dNQ4gKpXB7Q6kDdi18sEGRiZxcWRVAgZ?usp=sharing)  

### 👉 Đối Tượng Người Dùng  
- **Người dùng cá nhân**: Kiểm tra tin nhắn hoặc email đáng ngờ để bảo vệ thông tin cá nhân.  
- **Doanh nghiệp**: Tự động lọc email, giảm thiểu nguy cơ lừa đảo.  
- **Nhà nghiên cứu**: Tìm hiểu cách AI phân tích và giải thích dữ liệu tin nhắn.  

Bạn có thể nhập một tin nhắn để kiểm tra hoặc tích hợp hệ thống vào pipeline xử lý dữ liệu lớn.

---

## 🧠 Cách Hoạt Động  

Hệ thống sử dụng các kỹ thuật học máy và công nghệ hiện đại:  
- **Data Augmentation**: Tăng cường dữ liệu để cải thiện khả năng nhận diện spam tinh vi và tránh phân loại nhầm email có nội dung ham thành spam .  
- **Weighted KNN**: Phân loại dựa trên mức độ tương đồng, ưu tiên các mẫu gần nhất.  
- **Explainable AI (XAI)**: Sử dụng **Masking-based Saliency** để giải thích quyết định phân loại.  
- **HuggingFace + FAISS**: Tạo embedding ngữ nghĩa và tìm kiếm tin nhắn tương tự.  

Quy trình:  
1. Tin nhắn được mã hóa thành vector bằng mô hình `multilingual-e5-base`.  
2. Weighted KNN phân loại dựa trên dữ liệu huấn luyện.  
3. XAI phân tích và chỉ ra các từ khóa quan trọng ảnh hưởng đến kết quả.  

---

## 🔍 Tính Năng Nổi Bật  

- **Phân loại chính xác**: Gắn nhãn tin nhắn là spam hoặc ham.  
- **Giải thích rõ ràng**: Hiển thị các từ khóa và mức độ ảnh hưởng đến kết quả.  
- **Phân loại tiểu mục**: Xác định loại spam (quảng cáo, lừa đảo, v.v.).  
- **Trực quan hóa**: Hiển thị các tin nhắn tương tự và điểm tương đồng.  

---

## ♻️ Ưu Điểm  

- **Độ chính xác cao**: Nhờ Weighted KNN và dữ liệu tăng cường lên đến 94%
- **Minh bạch**: Giải thích AI giúp người dùng hiểu rõ quyết định.  
- **Linh hoạt**: Hỗ trợ song ngữ Tiếng Anh và Tiếng Việt và tích hợp dễ dàng.  
- **Ứng dụng thực tế**: Từ bảo vệ cá nhân đến tự động hóa doanh nghiệp.  

---

## 🛠️ Công Nghệ Sử Dụng  

| **Thành Phần**            | **Công Cụ**                                                                                           |  
|---------------------------|-------------------------------------------------------------------------------------------------------|  
| Mã nguồn                  | [Google Colab](https://colab.research.google.com/drive/1dNQ4gKpXB7Q6kDdi18sEGRiZxcWRVAgZ?usp=sharing)       |  
| Mô hình NLP               | [BERT](https://huggingface.co/bert-base-multilingual-cased)                                           |  
| Slide giới thiệu          | [Slide](https://www.canva.com/design/DAGu3xJo4fw/wHrvqq_Zcu1q6WCW-DIIBA/view)                         | 
| Embedding                 | `multilingual-e5-base`                                                                                |  
| Tìm kiếm tương đồng       | FAISS                                                                                                 |  
| Giải thích AI             | Masking-based Saliency                                                                                |  
| Phân loại tiểu mục        | Semi-supervised Learning                                                                              |  

---

📂 _Tài liệu đi kèm:_
{{< pdf src="/Time-Series-Team-Hub/pdf/M2W4D1_Spam_Detector.pdf" title="M2W4D1_Spam_Detector" height="700px" >}}

Hãy trải nghiệm hệ thống ngay hôm nay để khám phá cách AI bảo vệ bạn khỏi tin nhắn rác một cách thông minh và minh bạch. 

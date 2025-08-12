base_prompt_qa = """
	Based on the following context items, please answer the query about Ho Chi Minh's life and legacy.
	Give yourself room to think by extracting relevant passages from the context before answering the query.
	Don't return the thinking, only return the answer.
	Make sure your answers are as explanatory as possible.
	Use the following examples as reference for the ideal answer style:

	Example 1:
	User query: Who was Ho Chi Minh, and what were his major achievements?
	Answer: Ho Chi Minh was a revolutionary leader and the founding father of the Socialist Republic of Vietnam. He led the Vietnamese independence movement against French colonial rule and later against the United States during the Vietnam War. His major achievements include founding the Democratic Republic of Vietnam in 1945, leading the successful struggle for independence, and unifying North and South Vietnam under communist rule in 1975.

	Example 2:
	User query: What was Ho Chi Minh's role in the Vietnam War?
	Answer: Ho Chi Minh played a crucial role as the leader of North Vietnam during the Vietnam War. He directed the strategies for the North's resistance against the United States and South Vietnam, rallying support from both the North and international communist allies. Ho Chi Minh's leadership helped solidify North Vietnam's commitment to reunifying the country under communist rule, which ultimately succeeded after his death in 1969.

	Example 3:
	User query: How did Ho Chi Minh contribute to the independence movement in Vietnam?
	Answer: Ho Chi Minh's contributions to Vietnam's independence began in the early 20th century when he helped establish the Viet Minh in 1941. The Viet Minh aimed to unite the people of Vietnam to fight for independence from French colonial rule. He also gained significant international support, including from the Soviet Union and China, which aided in the success of Vietnam's independence in 1954.

	Example 4:
	User query: What was Ho Chi Minh's involvement in the 1945 August Revolution?
	Answer: In August 1945, Ho Chi Minh led the August Revolution, which resulted in the overthrow of the Japanese occupation forces and the establishment of the Democratic Republic of Vietnam. This event marked the successful culmination of years of struggle for independence, and Ho Chi Minh declared the country’s independence on September 2, 1945.

	User query: {query}
	Answer:
"""

generate_question_prompt = """
	Dựa trên các mục ngữ cảnh sau, vui lòng tạo một câu hỏi trắc nghiệm liên quan đến '{query}' về mã Python. Câu hỏi phải có một phần thân rõ ràng và bốn lựa chọn: một câu trả lời đúng và ba câu trả lời sai nhưng có vẻ hợp lý. Đảm bảo rằng câu trả lời đúng được hỗ trợ trực tiếp bởi ngữ cảnh, và các câu trả lời sai có liên quan đến chủ đề nhưng không đúng dựa trên ngữ cảnh.

	Hướng dẫn tạo câu hỏi:

		Xác định một sự thật hoặc thông tin chính từ ngữ cảnh liên quan đến '{query}' có thể được kiểm tra, chẳng hạn như mục đích của một hàm, giá trị của một biến hoặc đầu ra của một đoạn mã.

		Xây dựng phần thân câu hỏi một cách rõ ràng và cụ thể.

		Tạo bốn lựa chọn trong đó một lựa chọn là câu trả lời đúng, và ba lựa chọn còn lại là hợp lý nhưng không đúng.

		Đảm bảo rằng tất cả các lựa chọn có độ dài và định dạng tương tự nhau.

		Ngẫu nhiên hóa thứ tự các lựa chọn để câu trả lời đúng không phải lúc nào cũng ở cùng một vị trí.

	Trình bày câu hỏi của bạn theo định dạng này:

	Câu hỏi: [phần thân câu hỏi]
	Lựa chọn:
	A)[lựa chọn 1]
	B)[lựa chọn 2]
	C)[lựa chọn 3]
	D)[lựa chọn 4]
	Đáp án đúng: [chữ cái của lựa chọn đúng]

	Ví dụ về một câu hỏi trắc nghiệm tốt:
	Câu hỏi: Kết quả của print(2 + 3 * 4) trong Python sẽ là gì?
	Lựa chọn:
	A) 20
	B) 14
	C) 24
	D) 10
	Đáp án đúng: B
"""


mcq = """
        Bạn là một trợ lý chuyên tạo câu hỏi trắc nghiệm (MCQ).
        Mỗi câu hỏi gồm 1 câu hỏi (question), 4 lựa chọn, còn được gọi là choices (A, B, C, D), và chỉ 1 đáp án đúng, được gọi là correct.
        Đáp án đúng phải được đánh dấu rõ.
        Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết.

        Trả về output dưới dạng JSON duy nhất với đúng bốn khóa. Chỉ xuất ra đối tượng JSON, không thêm bất kỳ nội dung nào khác.

        Ví dụ về đầu ra JSON:
        {{
            "question": "...",
            "choices": {{
                "A": "...",
                "B": "...",
                "C": "...",
                "D": "..."
            }},
            "correct": "...",
            "explanation": "..."
        }}

        Context: {context}

        
        Hãy tạo một câu hỏi trắc nghiệm từ context.
"""
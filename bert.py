from transformers import pipeline

# 질의 응답을 위한 파이프라인 설정
qa_pipeline = pipeline("question-answering")

# 예시 컨텍스트 (리뷰 내용)
context = """
The game has several performance issues, including lag during high-action scenes.
Graphics are generally good, but frame drops are noticeable.
The control system is also unresponsive at times, affecting gameplay.
"""

# 질문에 대한 답을 찾기
questions = ["What performance issues are present?",
             "How are the graphics?",
             "What is the problem with controls?"]

for question in questions:
    answer = qa_pipeline(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {answer['answer']}\n")
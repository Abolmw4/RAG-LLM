from src.prompt_builder.prompt_builder import PromptBuilder
from langchain_core.documents import Document
from typing import List

class SimplePrompt(PromptBuilder):
    def __init__(self):
        super().__init__()
        pass
    
    def build_prompt(self, question: str, docs: List[Document]) -> str:
        if not question.strip():
            raise ValueError(f"question can not be emtpy")
        
        if not docs:
            raise ValueError(f"documents cannot be empty.")
        
        context = "\n\n".join(doc.page_content for doc in docs)
        
        prompt = f"""
        تو یک دستیار هوش مصنوعی هستی که وظیفه داری بر اساس متن و اطلاعاتی که در اختیارت قرار میگیرد پاسخ دقیق و واضح بدهی. اگر جواب توی متن و اطلاعاتی که بهت داده شده نبود خیلی واضح بگو نمیدانم اطلاعی ندارم.
        متن یا اطلاعات:
        {context}
        سوال:
        {question}
        پاسخ:
        """
        return prompt.strip()

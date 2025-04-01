from langchain_core.chat_history import BaseChatMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory

class CustomSQLMemory(BaseChatMemory):
    def __init__(self, user_id):
        self.chat_memory = SQLChatMessageHistory(
            session_id=user_id,
            connection="sqlite:///chat_history.db"
        )
    
    def load_memory_variables(self, inputs):
        return {"history": self.chat_memory.messages}
    
    def save_context(self, inputs, outputs):
        self.chat_memory.add_user_message(inputs['input'])
        self.chat_memory.add_ai_message(outputs['output'])

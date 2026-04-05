from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import CHROMA_PATH, EMBED_MODEL, STRICT_ORDER
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import os
import time
from config import (CHROMA_PATH, EMBED_MODEL, STRICT_ORDER, 
                    RAG_PROMPT_TEMPLATE, LLM_CONFIG)
from langchain_core.messages import HumanMessage, AIMessage

chat_history = [] # простейшее хранилище 

app = FastAPI(title="AI Research RAG API")
 
# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows browser to access the API
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Инициализация (делаем один раз при старте сервера)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
llm = OllamaLLM(**LLM_CONFIG)
# llm = OllamaLLM(
#     model="llama3",
#     model_kwargs={
#         "temperature": 0.4,
#         "repeat_penalty": 1.8, # as repetition_penalty
#         "top_p": 0.5,
#         "num_predict": 300, # as max_tokens (уменьшил ответ для быстроты)
#         "num_ctx": 4096, # context window for books chunks (2048 default)
#     }
# )

# Модель данных для запроса
class QuestionRequest(BaseModel):
    question: str
    level: str = "LLM set" # бывший user_choice


@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    # This serves the index.html file you created in the src folder
    ui_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(ui_path)


@app.get("/")
def read_root():
    return {"status": "AI Researcher is online", "modes": STRICT_ORDER}

@app.post("/ask")
async def ask_expert(request: QuestionRequest):
    global chat_history
    start_total = time.time()

    # RETRIEVAL
    # Логика фильтрации "Матрешки"
    target_tag = request.level.replace(" set", "")
    idx = STRICT_ORDER.index(target_tag)
    allowed_sets = [f"{t} set" for t in STRICT_ORDER[:idx+1]]

    start_search = time.time()
    # Поиск контекста
    docs = db.similarity_search(
        request.question, 
        k=3, 
        filter={"category_set": {"$in": allowed_sets}}
    )
    search_duration = time.time() - start_search

    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([f"{d.metadata.get('source')} [{d.metadata.get('category_set')}]" for d in docs]))
    
    # 2. Формируем историю для Llama-3
    # Берем последние 6 сообщений, чтобы не перегружать контекстное окно (num_ctx)
    history_context = chat_history[-6:]

    # 3. Обновляем промпт (добавляем блок истории)
    formatted_history = ""
    for msg in history_context:
        prefix = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted_history += f"{prefix}: {msg.content}\n"

    prompt = f"""
    {RAG_PROMPT_TEMPLATE.format(context=context, question=request.question)}
    
    PREVIOUS CONVERSATION:
    {formatted_history}
    """
    

    # Функция-генератор для стриминга
    def generate_tokens():
        start_gen = time.time()
        first_token_time = None
        full_response = "" # Собираем ответ, чтобы положить в историю

        # response 
        for chunk in llm.stream(prompt):
            if first_token_time is None:
                first_token_time = time.time() - start_gen
            full_response += chunk
            yield chunk

        # После завершения генерации сохраняем пару в историю
        chat_history.append(HumanMessage(content=request.question))
        chat_history.append(AIMessage(content=full_response))

        gen_duraction = time.time() - start_gen
        total_duration = time.time() - start_total

        # статистика/источники
        yield "\n\n---\n"
        yield f"📊 M1: Search: {search_duration:.2f}s | First: {first_token_time:.2f}s | Gen: {gen_duraction:.2f}s Total: {total_duration:.2f}s\n"
        yield f"🧠 History: {len(chat_history)} messages | Memory: Active\n"
        yield "📚 ИСТОЧНИКИ:\n"
        for s in sources:
            yield f"• {s}\n"

    return StreamingResponse(generate_tokens(), media_type="text/plain")

# clear chat memory
@app.post("/clear")
async def clear_memory():
    global chat_history
    chat_history = []
    return {"status": "Memory cleared"}
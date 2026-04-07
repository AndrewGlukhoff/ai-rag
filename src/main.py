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
                    RAG_PROMPT_BASIC, LLM_CONFIG, RAG_PROMPT_COT)
from langchain_core.messages import HumanMessage, AIMessage
import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


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

LLM_CONFIG.update({
    "temperature": 0.1,
    # Оставляем только английские маркеры, чтобы не путать русский поиск
    "stop": ["Summary:", "Resumen:", "Translation:", "Assistant:"] 
})

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
    level: str = "LLM set" 
    deep_think: bool = False


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
    base_filter = {"category_set": {"$in": allowed_sets}}

    detected_authors = get_automatic_filters(request.question, chat_history)
    if detected_authors:
        print(f"🎯 SURGICAL STRIKE: Фильтр по авторам: {detected_authors}")
        # filter
        author_exprs = [{"author": {"$eq": s}} for s in detected_authors]
        if len(author_exprs) == 1:
            chroma_filter = {
                "$and": [
                    base_filter,
                    author_exprs[0] # Use the single expression directly
                ]
            }
        else: 
            chroma_filter = {
                "$and": [
                    base_filter,
                    {"$or": author_exprs}
                ]
            }
    else:
        print("🔍 ОБЩИЙ ПОИСК: Совпадений по авторам не найдено.")
        chroma_filter = base_filter


    start_search = time.time()
    # surgical strike authors
    docs = db.similarity_search(
        request.question, 
        k=7, # увеличил чтобы попало больше книжек для сравнения
        filter=chroma_filter
    )

    # доп/поиск на случай вдруг мы залипли на конкретном авторе из пред/контекста
    global_docs = db.similarity_search(
        request.question,
        k=4,
        filter=base_filter
    )

    all_docs = docs + global_docs
    # take unique only
    seen_content = set()
    unique_docs = []
    for d in all_docs:
        if d.page_content not in seen_content:
            unique_docs.append(d)
            seen_content.add(d.page_content)
    docs = unique_docs[:10] # take 10

    search_duration = time.time() - start_search
    print(f"DEBUG: Found {len(docs)} chunks from DB.")
    if len(docs) > 0:
        print(f"DEBUG: Sample source from first chunk: {docs[0].metadata['source']}")

    context_parts = []
    for d in docs:
        source_name = d.metadata.get('source', 'Unknown')
        author_name = d.metadata.get('author', 'Unknown')
        book_lang = d.metadata.get('lang', 'eng')

        # Добавляем перед каждым чанком
        header = f"--- ИСТОЧНИК: {source_name} | АВТОР: {author_name} | ЯЗЫК: {book_lang} ---"
        content = f"{header}\n{d.page_content}"
        context_parts.append(content)
    
    context = "\n\n".join(context_parts)

    sources = list(set([f"{d.metadata.get('source')} [{d.metadata.get('category_set')}]" for d in docs]))
    
    # Берем последние 6 сообщений, чтобы не перегружать контекстное окно (num_ctx)
    history_context = chat_history[-6:]

    # Choose the prompt based on the checkbox
    instruction = RAG_PROMPT_COT if request.deep_think else RAG_PROMPT_BASIC
    system_instruction = f"""
    Ты — технический эксперт. {instruction} 
    ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ. 
    ЗАПРЕЩЕНО ДУБЛИРОВАТЬ ОТВЕТ НА АНГЛИЙСКОМ. 
    ПИШИ СРАЗУ ПО-РУССКИ.
    """
    messages = [SystemMessage(content=system_instruction)]
    messages.extend(chat_history[-6:])
    
    current_user_content = f"КОНТЕКСТ ИЗ КНИГ:\n{context}\n\nВОПРОС: {request.question}"
    messages.append(HumanMessage(content=current_user_content))

    if request.deep_think:
        messages.append(AIMessage(content="<thought>"))

    # Функция-генератор для стриминга
    def generate_tokens():
        start_gen = time.time()
        first_token_time = None
        full_response = "" # Собираем ответ, чтобы положить в историю

        # response 
        for chunk in llm.stream(messages):
            if first_token_time is None:
                first_token_time = time.time() - start_gen
            full_response += chunk
            yield chunk

        # Сохраняем в историю ТОЛЬКО чистый ответ модели
        chat_history.append(HumanMessage(content=request.question))
        chat_history.append(AIMessage(content=full_response))

        gen_duraction = time.time() - start_gen
        total_duration = time.time() - start_total

        # статистика/источники
        stats_block = (
            f"\n\n---\n"
            f"📊 M1: Search: {search_duration:.2f}s | First: {first_token_time:.2f}s | Gen: {gen_duraction:.2f}s Total: {total_duration:.2f}s\n"
            f"🧠 History: {len(chat_history)} messages | Memory: Active\n"
            f"📚 ИСТОЧНИКИ:\n" + "\n".join([f"• {s}" for s in sources])
        )
        yield stats_block

    return StreamingResponse(generate_tokens(), media_type="text/plain")

# clear chat memory
@app.post("/clear")
async def clear_memory():
    global chat_history
    chat_history = []
    print("🧹 Chat history has been wiped cleaner than a whistle!")
    return {"status": "success", "message": "Memory cleared"}


# Глобальный список всех файлов в базе
UNIQUE_SOURCES = []
UNIQUE_AUTHORS = []

@app.on_event("startup")
async def startup_event():
    global UNIQUE_SOURCES, UNIQUE_AUTHORS
    try:
        print("🔍 Синхронизация с ChromaDB...")
        all_sources = set()
        all_authors = set()
        
        total_count = db._collection.count()
        batch_size = 1000 
        
        for i in range(0, total_count, batch_size):
            data = db.get(
                include=['metadatas'],
                limit=batch_size,
                offset=i
            )
            if data and data['metadatas']:
                for m in data['metadatas']:
                    if m:
                        if m.get('source'):
                            all_sources.add(m.get('source'))
                        author = m.get('author')
                        if author and author != 'Unknown':
                            all_authors.add(author)
        
        UNIQUE_SOURCES = list(all_sources)
        UNIQUE_AUTHORS = list(all_authors)
        
        print(f"✅ Синхронизация окончена!")
        print(f"📚 Книг: {len(UNIQUE_SOURCES)} | ✍️ Авторов: {len(UNIQUE_AUTHORS)}")
        print(f"🧩 Всего чанков в базе: {total_count}")
        
    except Exception as e:
        print(f"❌ Error during source scan: {e}")


import re

def get_automatic_filters(user_query: str, history: list):
    print(f"DEBUG: len(history): {len(history)}")
    
    search_context = user_query.lower()
    for msg in history[-2:]:
        search_context += " " + msg.content.lower()

    # TODO подумать включить транслитерацию
    
    matched_authors = set()
    
    STOP_WORDS = {} 
    for author in UNIQUE_AUTHORS:
        # Разбиваем сложные имена "John Berryman & Albert Ziegler" на части
        name_parts = re.split(r'[&\s,]', author.lower())
        
        for part in name_parts:
            if len(part) > 4 and part not in STOP_WORDS:
                if part in search_context:
                    print(f"DEBUG: part: {part}, name_parts: {name_parts}")
                    print(f"DEBUG: search_context: {search_context}")
                    matched_authors.add(author)
                    break 
                
    return list(matched_authors)


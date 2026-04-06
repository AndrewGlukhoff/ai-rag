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


    # Автоматически находим нужные файлы по фамилиям в вопросе
    matched_sources = get_automatic_filters(request.question, chat_history)
    
    base_filter = {"category_set": {"$in": allowed_sets}}
    
    if matched_sources:
        print(f"🎯 SURGICAL STRIKE: Найдено совпадений: {len(matched_sources)}")
        print(f"📋 Источники: {matched_sources}")

        source_filters = [{"source": {"$eq": s}} for s in matched_sources]
        if len(source_filters) == 1:
            chroma_filter = {
                "$and": [
                    base_filter,
                    source_filters[0] # Use the single expression directly
                ]
            }
        else: 
            chroma_filter = {
                "$and": [
                    base_filter,
                    {"$or": source_filters}
                ]
            }
    else:
        print("🔍 ОБЩИЙ ПОИСК: Совпадений по авторам не найдено.")
        chroma_filter = base_filter


    start_search = time.time()
    docs = db.similarity_search(
        request.question, 
        k=7, # увеличил чтобы попало больше книжек для сравнения
        filter=chroma_filter
    )
    search_duration = time.time() - start_search
    print(f"DEBUG: Found {len(docs)} chunks from DB.")
    if len(docs) > 0:
        print(f"DEBUG: Sample source from first chunk: {docs[0].metadata['source']}")


    # context = "\n\n".join([d.page_content for d in docs])
    # именованный контекст
    context_parts = []
    for d in docs:
        source_name = d.metadata.get('source', 'Unknown')
        # Добавляем имя источника перед каждым чанком
        content = f"--- ИСТОЧНИК: {source_name} ---\n{d.page_content}"
        context_parts.append(content)
    
    context = "\n\n".join(context_parts)

    sources = list(set([f"{d.metadata.get('source')} [{d.metadata.get('category_set')}]" for d in docs]))
    
    # 2. Формируем историю для Llama-3
    # Берем последние 6 сообщений, чтобы не перегружать контекстное окно (num_ctx)
    history_context = chat_history[-6:]

    # # Обновляем промпт (добавляем блок истории)
    # formatted_history = ""
    # for msg in history_context:
    #     prefix = "User" if isinstance(msg, HumanMessage) else "Assistant"
    #     formatted_history += f"{prefix}: {msg.content}\n"


    # Choose the prompt based on the checkbox
    instruction = RAG_PROMPT_COT if request.deep_think else RAG_PROMPT_BASIC
    
#     prompt = f"""
#     SYSTEM: Ты — РУССКОЯЗЫЧНЫЙ ИССЛЕДОВАТЕЛЬ ИИ. Твой единственный источник — предоставленный КОНТЕКСТ. 
#     ЗАПРЕЩЕНО ИСПОЛЬЗОВАТЬ АНГЛИЙСКИЙ ЯЗЫК ДЛЯ РАССУЖДЕНИЙ.

#     ИСТОРИЯ ДИАЛОГА:
#     {formatted_history}

#     КОНТЕКСТ ИЗ КНИГ:
#     {context}
   
#     ЗАДАЧА: 
#     {instruction}

#     ВОПРОС: {request.question}

#     ОБЯЗАТЕЛЬНО НАЧНИ С ТЕГА <thought> И ПИШИ ТОЛЬКО ПО-РУССКИ:
# """
    # if request.deep_think:
    #     prompt += "<thought>"

    messages = [SystemMessage(content=f"{instruction}\nОТВЕЧАЙ ТОЛЬКО НА РУССКОМ.")]
    
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

@app.on_event("startup")
async def startup_event():
    global UNIQUE_SOURCES
    try:
        print("🔍 Scanning ChromaDB for unique sources...")
        all_sources = set()
        
        # 1. Get the total count of items in the DB
        total_count = db._collection.count()
        batch_size = 1000 # SQLite friendly size
        
        # 2. Fetch metadata in batches to avoid "too many variables" error
        for i in range(0, total_count, batch_size):
            data = db.get(
                include=['metadatas'],
                limit=batch_size,
                offset=i
            )
            if data and data['metadatas']:
                batch_sources = [m.get('source') for m in data['metadatas'] if m]
                all_sources.update(batch_sources)
        
        UNIQUE_SOURCES = list(all_sources)
        print(f"✅ Indexed {len(UNIQUE_SOURCES)} unique books/sources from {total_count} chunks.")
        
    except Exception as e:
        print(f"❌ Error during source scan: {e}")


import re

def get_automatic_filters(user_query: str, history: list):
    query_lower = user_query.lower()
    print(f"DEBUG: len(history): {len(history)}")
    for msg in history[-2:]:
        query_lower += " " + msg.content.lower()
    
    matched = set()
    # TODO вместо этого фильтра сделать новый ingest с отдельно authors
    STOP_WORDS = {"engineering", "machine", "learning", "artificial", "intelligence", "large", "language", "prompt"} 
    for source in UNIQUE_SOURCES:
        clean_name = re.sub(r'\.(epub|pdf|pdf_txt)$', '', source, flags=re.IGNORECASE)
        name_parts = re.split(r'[-_\s\.]', clean_name.lower())
        
        for part in name_parts:
            # Ищем фамилии (длиной > 4) в вопросе ИЛИ в истории
            if len(part) > 4 and part in query_lower and part not in STOP_WORDS:
                print(f"DEBUG: part: {part}, name_parts: {name_parts}")
                print(f"DEBUG: query_lower: {query_lower}")
                matched.add(source)
                break 
                
    return list(matched)


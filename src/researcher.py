import os
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import CHROMA_PATH, EMBED_MODEL, STRICT_ORDER

# 1. Подключаемся к Ollama (модель из ~/.ollama/)
print("🧠 Подключение к Ollama Llama-3...")
llm = OllamaLLM(
    model="llama3",
    model_kwargs={
        "temperature": 0.4,
        "repeat_penalty": 1.8, # as repetition_penalty
        "top_p": 0.5,
        "num_predict": 400, # as max_tokens
        "num_ctx": 4096, # context window for books chunks (2048 default)
    }
)

# 2. Инициализируем поиск по твоим 7 книгам
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def ask_expert(question, user_choice="LLM set"):
    print(f"🔎 Исследую вопрос: '{question}'...")

    # Определяем, какие сеты мы разрешаем видеть
    # Если юзер выбрал "DL set", он видит ["LLM set", "NLP set", "DL set"]
    target_tag = user_choice.replace(" set", "")
    idx = STRICT_ORDER.index(target_tag)
    allowed_sets = [f"{t} set" for t in STRICT_ORDER[:idx+1]]
    
    print(f"🔎 Уровень: {user_choice}. Ищем в категориях: {allowed_sets}")
    
    # Ищем k самых точных куска текста в базе
    docs = db.similarity_search(
        question, 
        k=5, 
        filter={"category_set": {"$in": allowed_sets}}
    )

    context = "\n\n".join([d.page_content for d in docs])
    
    # Формируем строгий промпт для RAG
    prompt = f"""
    ### ИНСТРУКЦИЯ ДЛЯ ТЕХНИЧЕСКОГО ЭКСПЕРТА ###
    Ты — ведущий исследователь ИИ. Твоя задача: на основе предоставленных отрывков из технической литературы (на английском) составить ГЛУБОКИЙ и ПОДРОБНЫЙ ответ на русском языке.

    ПРАВИЛА:
    1. Используй профессиональную терминологию (LLM, токены, контекстное окно, веса).
    2. Не давай общих определений, если в тексте есть конкретика (авторы, методы, примеры).
    3. Сгруппируй информацию логически: определение, методология, важность.
    4. Отвечай только на основе предоставленного ТЕКСТА.

    КОНТЕКСТ (из твоей библиотеки):
    {context}

    ВОПРОС: {question}

    ПОДРОБНЫЙ ОТВЕТ НА РУССКОМ:
"""

    # Генерируем ответ через Ollama
    response = llm.invoke(prompt)
    
    print("\n🎓 ОТВЕТ АССИСТЕНТА-ИССЛЕДОВАТЕЛЯ:")
    print(response)
    
    # Показываем, откуда взята информация + категория
    source_info = []
    for d in docs:
        src = d.metadata.get('source', 'Unknown')
        cat = d.metadata.get('category_set', 'No Cat')
        source_info.append(f"{src} [{cat}]")

    unique_sources = list(set(source_info))
    print(f"\n📚 SOURCES USED:\n" + "\n".join([f"- {s}" for s in unique_sources]))

    # sources = set([os.path.basename(d.metadata.get('source', 'Книга')) for d in docs])
    # print(f"\n📚 ИСТОЧНИКИ: {', '.join(sources)}")



def select_expertise():
    print("\n--- CHOOSE EXPERTISE LEVEL: 1(LLM), 2(NLP), 3(DL), 4(ML), 5(AI) ---")
    for i, level in enumerate(STRICT_ORDER):
        print(f"{i+1}. {level} set")
    
    choice = input("\nEnter number (default 1 - LLM): ")
    if not choice or not choice.isdigit():
        return "LLM set"
    
    idx = int(choice) - 1
    if 0 <= idx < len(STRICT_ORDER):
        return f"{STRICT_ORDER[idx]} set"
    return "LLM set"




if __name__ == "__main__":
    question = "Какие конкретные методы промпт-инжиниринга ты знаешь для уменьшения галлюцинаций?"

    selected_level = select_expertise()
    ask_expert(question, user_choice=selected_level)

import os
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import CHROMA_PATH, EMBED_MODEL, STRICT_ORDER, RAG_PROMPT_TEMPLATE, LLM_CONFIG
import time

# 1. Подключаемся к Ollama (модель из ~/.ollama/)
print("🧠 Подключение к Ollama Llama-3...")
llm = OllamaLLM(**LLM_CONFIG)

# 2. Инициализируем поиск по твоим 7 книгам
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def ask_expert(question, user_choice="LLM set"):
    start_total = time.time()
    print(f"🔎 Исследую вопрос: '{question}'...")

    # RETRIEVAL
    # Определяем, какие сеты мы разрешаем видеть
    # Если юзер выбрал "DL set", он видит ["LLM set", "NLP set", "DL set"]
    target_tag = user_choice.replace(" set", "")
    idx = STRICT_ORDER.index(target_tag)
    allowed_sets = [f"{t} set" for t in STRICT_ORDER[:idx+1]]
    
    print(f"🔎 Уровень: {user_choice}. Ищем в категориях: {allowed_sets}")
    start_search = time.time()
    
    # Ищем k самых точных куска текста в базе
    docs = db.similarity_search(
        question, 
        k=3, # уменьшил для быстроты 
        filter={"category_set": {"$in": allowed_sets}}
    )

    search_duration = time.time() - start_search
    print(f"⏱️ Search took: {search_duration:.2f} sec")

    if not docs:
        print("❌ No context found!")
        return

    # GENERATION
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    # Генерируем ответ через Ollama    
    print("\n🎓 ОТВЕТ АССИСТЕНТА-ИССЛЕДОВАТЕЛЯ:")
    start_gen = time.time()

    # используем поток для быстроты
    for chunk in llm.stream(prompt):
        print(chunk, end="", flush=True)
    print("\n")

    gen_duration = time.time() - start_gen
    total_duration = time.time() - start_total

    print(f"\n\n{'='*30}")
    print(f"📊 PERF REPORT (M1 16GB):")
    print(f"   - Retrieval: {search_duration:.2f}s")
    print(f"   - Generation: {gen_duration:.2f}s")
    print(f"   - Total: {total_duration:.2f}s")
    
    # Показываем, откуда взята информация + категория
    source_info = list(set([f"{d.metadata.get('source')} [{d.metadata.get('category_set')}]" for d in docs]))
    print(f"📚 SOURCES:\n" + "\n".join([f"   - {s}" for s in source_info]))

    
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

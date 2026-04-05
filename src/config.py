# src/config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# REMOTE (Acer Laptop via SMB)
CALIBRE_PATH = "/Volumes/Calibre2" 
DB_FILE = os.path.join(CALIBRE_PATH, "metadata.db")

# LOCAL (M1 SSD)
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" #"sentence-transformers/all-MiniLM-L6-v2" 

# CHUNK SETTINGS
CHUNK_SIZE = 1500 # для тех/книг мало 700 символов.
CHUNK_OVERLAP = 150 # 70

# Mapping of Calibre Tags to RAG Categories
CATEGORY_HIERARCHY = {
    "LLM set": ["LLM"],
    "NLP set": ["NLP", "LLM"],
    "DL set":  ["DL", "NLP", "LLM"],
    "ML set":  ["ML", "DL", "NLP", "LLM"],
    "AI set":  ["AI", "ML", "DL", "NLP", "LLM"]
}

# Иерархия от частного к общему (чем левее, тем строже)
STRICT_ORDER = ["LLM", "NLP", "DL", "ML", "AI"]

def get_best_category_set(calibre_tags_str: str) -> str:
    """
    Выбирает самый строгий тег из доступных.
    Пример: 'AI, IT, NLP, Java' -> 'NLP set'
    """
    if not calibre_tags_str:
        return "AI set"
        
    # Чистим теги из Calibre
    tags = [t.strip().upper() for t in calibre_tags_str.split(",")]
    
    # Ищем первое совпадение в порядке строгости
    for s_tag in STRICT_ORDER:
        if s_tag in tags:
            return f"{s_tag} set"
            
    return "AI set" # Дефолт, если совпадений нет

# в этом проекте отсутствуют спец/токены (как в chatbot), т.к. ollama позаботится об этом

# Короткая инструкция
RAG_PROMPT_BASIC = "Проанализируй контекст и кратко ответь на вопрос пользователя на русском языке."

# Глубокая инструкция (CoT)
RAG_PROMPT_COT = """
### СТРОГИЙ ТЕХНИЧЕСКИЙ АУДИТОР ###
Твоя главная задача: проверить наличие фактов в КОНТЕКСТЕ. 

ПРАВИЛА:
1. В блоке <thought> начни с фразы: "Проверяю наличие [название термина] в предоставленном тексте..."
2. Если в КОНТЕКСТЕ нет упоминания конкретного алгоритма или автора, о которых спрашивает юзер — ТЫ ОБЯЗАН СКАЗАТЬ: "В предоставленных материалах информация об этом отсутствует".
3. ЗАПРЕЩЕНО выдумывать или использовать внешние знания, если их нет в тексте.

СТРУКТУРА:
<thought>
(Анализ: есть ли этот термин в тексте? На каких страницах/абзацах? Если нет — почему?)
</thought>

ИТОГОВЫЙ ОТВЕТ:
(Только факты из текста или признание отсутствия информации)
"""


LLM_CONFIG = {
    "model": "llama3",
    "temperature": 0.4,
    "repeat_penalty": 1.2,
    "top_p": 0.5,
    "num_predict": 800, # для CoT 300 будет мало
    "num_ctx": 4096,
}
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




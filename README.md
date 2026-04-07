## 📚 M1 Research RAG:  AI Librarian

Интеллектуальный ассистент локальной библиотеки (96+ книг), развернутый на **Apple M1 (16GB)**. Система использует RAG-архитектуру с  фильтрацией метаданных из **Calibre**.

**Ключевые возможности**

- **Surgical Search**: Автоматическое распознавание имен авторов и фильтрация базы ChromaDB по конкретным источникам.
- **Onion Brain Strategy**: Многоуровневая фильтрация контекста (LLM set, ML set, AI set) для отсечения шума.
- **Deep Think (CoT)**: Режим пошагового рассуждения модели для сложного анализа и сравнения авторов.
- **Bilingual Support**: Поиск по английским книгам с использованием русских и английских запросов.
- **Calibre Integration**: Прямая синхронизация метаданных (автор, язык, теги) из `metadata.db` Calibre.

**Технический стек**

- **LLM**: Llama-3 8B (via Ollama)
- **Vector DB**: ChromaDB (64,000+ чанков)
- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2
- **Backend**: FastAPI (Python 3.11)
- **Frontend**: Vanilla JS + Streaming API (SSE)

**Установка и запуск**

```
git clone https://github.com
cd ai_library_rag
```


**Окружение** 
```
# update conda
conda update -n base -c defaults conda

# 1. Создаем чистое окружение Python 3.11
conda create -n ai_rag python=3.11 -y
conda activate ai_rag

# 2. Устанавливаем базу для MLX (нативный движок Apple)
pip install mlx-lm

# 3. Устанавливаем RAG-стек
# langchain-community: обертки для загрузчиков 
# chromadb: векторная база (хранится локально)
# sentence-transformers: для перевода текста в векторы (Embeddings)
pip install langchain langchain-community chromadb sentence-transformers

# 4. Библиотеки для работы с форматами (EPUB, PDF, SQLite)
# unstructured[epub]: для чтения книг
# pypdf: легкий загрузчик PDF
pip install "unstructured[epub]" pypdf 

# 5. Утилиты
pip install pandas psutil pyyaml

# связать env ai_rag и kernels
pip install ipykernel
python -m ipykernel install --user --name ai_rag --display-name "Python 3.11 (ai_rag)"
# --user: только для текущего юзера
# --name: внутр/имя
# --display-name: имя ядра
# справа в статус строке select 1)interpreter 2)python env 
# check Settings -> Search: python.defaultInterpreterPath
# должна быть как результат $ which python #e.g. /opt/anaconda3/envs/ai_rag/bin/python
# Cmd + Shift + P -> Restart Language Server

# ollama API
pip install -U langchain-ollama

# Fast API, uvicorn
pip install fastapi uvicorn pydantic
```

**Индексация библиотеки**
```
python src/ingest.py # path to Calibre folders in config.py

```

**Запуск сервера**
```
python uvicorn src.main:app --reload
```

**Browser**
```
http://127.0.0.1:8000/ui
```
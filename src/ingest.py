import os
import time
import sqlite3
from config import CALIBRE_PATH, DB_FILE
from config import get_best_category_set, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PATH, EMBED_MODEL


def wait_for_connection(path, check_interval=10):
    """Ставит скрипт на паузу, пока сетевой диск не станет доступен."""
    while not os.path.exists(path):
        print(f"⚠️ Сетевой диск {path} недоступен. Acer ушел в сон?")
        print(f"Ожидание {check_interval} секунд...")
        time.sleep(check_interval)
    print("✅ Соединение восстановлено. Продолжаем...")

import sqlite3
import os
from config import CALIBRE_PATH, get_best_category_set

def get_books_with_tags(db_path, allowed_tags=["AI", "ML", "DL", "NLP", "LLM"]):
    """
    Fetches book paths and their combined tags, authors, languages from Calibre.
    """
    # SQL to get: Path, Name, and a comma-separated list of all Tags for that book
    query = """
    SELECT b.path, d.name, d.format, 
           (SELECT GROUP_CONCAT(t.name) FROM tags t 
            JOIN books_tags_link btl ON t.id = btl.tag 
            WHERE btl.book = b.id) as all_tags,
           (SELECT GROUP_CONCAT(a.name, ' & ') FROM authors a 
            JOIN books_authors_link bal ON a.id = bal.author 
            WHERE bal.book = b.id) as all_authors, 
           (SELECT l.lang_code FROM languages l 
            JOIN books_languages_link bll ON l.id = bll.lang_code 
            WHERE bll.book = b.id LIMIT 1) as lang 
    FROM books b
    JOIN data d ON b.id = d.book
    WHERE d.format IN ('EPUB', 'PDF')
    """
    
    books_data = []
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        for row in cursor.fetchall():
            rel_path, file_name, extension, all_tags, all_authors, lang = row
            
            # Filter: only keep books that have at least one of our target tags
            if all_tags:
                tag_list = [t.strip().upper() for t in all_tags.split(",")]
                if any(tag in tag_list for tag in allowed_tags):
                    full_path = os.path.join(CALIBRE_PATH, rel_path, f"{file_name}.{extension.lower()}")
                    books_data.append({
                        "path": full_path,
                        "tags": all_tags,
                        "author": all_authors if all_authors else "Unknown",
                        "lang": lang if lang else "English"
                    })
                    
    return books_data

from langchain_community.document_loaders import PyPDFLoader, UnstructuredEPubLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import CHROMA_PATH, EMBED_MODEL, CHUNK_OVERLAP, CHUNK_SIZE

def process_books(books_data):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    all_chunks = []

    total_books = len(books_data) # Всего книг
    print(f"🚀 Начинаем индексацию {total_books} книг...")
    
    for index, book in enumerate(books_data,1):
        path = book["path"]
        tags = book["tags"]
        author = book["author"]
        lang = book["lang"]

        percent = (index / total_books) * 100
        
        print(f"\n[{percent:.1f}%] — Книга {index} из {total_books}")
        print(f"📖 Processing: {os.path.basename(path)}")
        print(f"   Author: {author} | Lang: {lang}")

        # 1. Determine the "Elite" Set for this specific book
        best_set = get_best_category_set(tags)
        print(f"📖 Processing: {os.path.basename(path)}")
        print(f"   Tags: [{tags}] -> Assigned to: [{best_set}]")
        print(f"   Author: {author} | Lang: {lang}")
        
        wait_for_connection(path) # Safety check for Acer laptop
        
        try:
            # 2. Load
            loader = PyPDFLoader(path) if path.endswith(".pdf") else UnstructuredEPubLoader(path)
            
            # 3. Split (Using the CHUNK_SIZE chunk size)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP
            )
            docs = loader.load_and_split(splitter)
            
            # 4. Attach Metadata
            for doc in docs:
                doc.metadata["category_set"] = best_set
                doc.metadata["source"] = os.path.basename(path)
                doc.metadata["author"] = author
                doc.metadata["lang"] = lang
                
            all_chunks.extend(docs)
            print(f"   ✅ Created {len(docs)} chunks.")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # 5. Save to local M1 ChromaDB
    if all_chunks:
        print(f"💾 Saving {len(all_chunks)} chunks to local ChromaDB...")
        vector_db = Chroma.from_documents(
            documents=all_chunks, 
            embedding=embeddings, 
            persist_directory=CHROMA_PATH
        )
        print("✨ Ingestion Complete! Your 'Onion' Brain is ready.")


# Запуск процесса
books_data = get_books_with_tags(DB_FILE)
if books_data:
    process_books(books_data)

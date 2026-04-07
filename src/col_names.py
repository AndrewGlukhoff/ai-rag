import sqlite3
from config import CALIBRE_PATH
import os

DB_FILE = os.path.join(CALIBRE_PATH, "metadata.db")
conn = sqlite3.connect(DB_FILE) 
cursor = conn.cursor()
# cursor.execute("PRAGMA table_info(languages)") # [(0, 'id', 'INTEGER', 0, None, 1), (1, 'lang_code', 'TEXT NON', 0, None, 0), (2, 'link', 'TEXT', 1, "''", 0)]
cursor.execute("PRAGMA table_info(books_languages_link)") # [(0, 'id', 'INTEGER', 0, None, 1), (1, 'book', 'INTEGER', 1, None, 0), (2, 'lang_code', 'INTEGER', 1, None, 0), (3, 'item_order', 'INTEGER', 1, '0', 0)]
print(cursor.fetchall()) 
conn.close()

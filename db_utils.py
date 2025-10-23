import sqlite3
from datetime import datetime

DB_NAME="chat_history.db"

def get_db_connection():
    conn=sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

#create the chat hsitory table
def create_chat_table():
    conn=get_db_connection()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chat_history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.close()

#save chat history
def save_chat(question, answer):
    conn=get_db_connection()
    conn.execute('INSERT INTO chat_history (question, answer) VALUES (?, ?)',
                 (question, answer))
    conn.commit()
    conn.close()

#get recent chat history
def get_recent_history(limit=3):
    conn=get_db_connection()
    cursor=conn.cursor()
    cursor.execute('SELECT * FROM chat_history ORDER BY created_at DESC LIMIT ?', (limit,))
    rows=cursor.fetchall()
    conn.close()
    
    #return in chronological order(oldest fisrt)
    history=[]
    for row in reversed(rows):
        history.append({
            "question": row[1],
            "answer": row[2],
        })
    return history

#clear all chat history
def clear_history():
    conn=get_db_connection()
    conn.execute('DELETE FROM chat_history')
    conn.commit()
    conn.close()

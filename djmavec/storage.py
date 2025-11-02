# djmavec/storage.py

import json
import mysql.connector
from typing import Dict, List, Any
import os

# --- Database Configuration ---

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "Hari@2005")
DB_NAME = "djmavec_db"
TABLE_NAME = "vectors"

_db_connection = None

def _get_db_connection():
    """Establishes and returns a database connection."""
    global _db_connection
    if _db_connection and _db_connection.is_connected():
        return _db_connection
    
    try:
        _db_connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = _db_connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        cursor.execute(f"USE {DB_NAME}")
        
        # Create table if it doesn't exist
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id VARCHAR(255) PRIMARY KEY,
            text TEXT,
            embedding JSON
        );
        """
        cursor.execute(create_table_query)
        _db_connection.commit()
        cursor.close()
        
        return _db_connection
    except mysql.connector.Error as err:
        print(f"Error connecting to MariaDB: {err}")
        print("Please ensure MariaDB is running and the credentials are correct.")
        print("You can set DB_HOST, DB_USER, and DB_PASSWORD environment variables.")
        return None

def save_record(record_id: str, text: str, embedding: List[float]):
    """Saves or updates a record in the database."""
    conn = _get_db_connection()
    if not conn:
        return

    cursor = conn.cursor()
    embedding_json = json.dumps(embedding)
    query = f"""
    INSERT INTO {TABLE_NAME} (id, text, embedding)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE text = VALUES(text), embedding = VALUES(embedding);
    """
    try:
        cursor.execute(query, (record_id, text, embedding_json))
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Failed to save record: {err}")
    finally:
        cursor.close()

def load_all_records() -> List[Dict[str, Any]]:
    """Loads all records from the database."""
    conn = _get_db_connection()
    if not conn:
        return []

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(f"SELECT id, text, embedding FROM {TABLE_NAME}")
        records = cursor.fetchall()
        for r in records:
            if isinstance(r['embedding'], str):
                r['embedding'] = json.loads(r['embedding'])
        return records
    except mysql.connector.Error as err:
        print(f"Failed to load records: {err}")
        return []
    finally:
        cursor.close()

def clear_db():
    """Clears all records from the table."""
    conn = _get_db_connection()
    if not conn:
        return
        
    cursor = conn.cursor()
    try:
        cursor.execute(f"TRUNCATE TABLE {TABLE_NAME}")
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Failed to clear table: {err}")
    finally:
        cursor.close()

"""
Migration script to add missing columns to medical_document table
and create missing tables for document management features.
"""
import sqlite3
import os

def migrate():
    db_path = 'ivf_tracker.db'
    if not os.path.exists(db_path):
        db_path = 'instance/ivf_tracker.db'

    print(f"Using database: {os.path.abspath(db_path)}")
    print(f"Database exists: {os.path.exists(db_path)}")
    if os.path.exists(db_path):
        print(f"Database size: {os.path.getsize(db_path)} bytes")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables in database: {[t[0] for t in tables]}")

    # Check current columns in medical_document
    cursor.execute("PRAGMA table_info(medical_document)")
    existing_cols = {col[1] for col in cursor.fetchall()}
    print(f"Existing columns in medical_document: {existing_cols}")

    # Add missing columns to medical_document
    new_columns = {
        'title': 'VARCHAR(255)',
        'category': "VARCHAR(50) DEFAULT 'other'",
        'version': 'INTEGER DEFAULT 1',
        'uploaded_by': "VARCHAR(10) DEFAULT 'patient'",
        'tags': 'VARCHAR(255)'
    }

    for col_name, col_type in new_columns.items():
        if col_name not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE medical_document ADD COLUMN {col_name} {col_type}")
                print(f"Added column: {col_name} ({col_type})")
            except sqlite3.OperationalError as e:
                print(f"Error adding {col_name}: {e}")
        else:
            print(f"Column already exists: {col_name}")

    # Create missing tables
    tables_to_create = {
        'document_version': """
            CREATE TABLE IF NOT EXISTS document_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                version INTEGER NOT NULL,
                filename VARCHAR(255) NOT NULL,
                uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                change_note TEXT,
                FOREIGN KEY (document_id) REFERENCES medical_document (id)
            )
        """,
        'document_annotation': """
            CREATE TABLE IF NOT EXISTS document_annotation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                annotation_text TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES medical_document (id),
                FOREIGN KEY (doctor_id) REFERENCES user (id)
            )
        """,
        'document_share': """
            CREATE TABLE IF NOT EXISTS document_share (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                shared_with_user_id INTEGER NOT NULL,
                shared_by_user_id INTEGER NOT NULL,
                shared_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                can_view BOOLEAN DEFAULT 1,
                FOREIGN KEY (document_id) REFERENCES medical_document (id),
                FOREIGN KEY (shared_with_user_id) REFERENCES user (id),
                FOREIGN KEY (shared_by_user_id) REFERENCES user (id)
            )
        """,
        'audit_log': """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action VARCHAR(50) NOT NULL,
                details VARCHAR(255),
                ip_address VARCHAR(50),
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user (id)
            )
        """
    }

    for table_name, create_sql in tables_to_create.items():
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            cursor.execute(create_sql)
            print(f"Created table: {table_name}")
        else:
            print(f"Table already exists: {table_name}")

    conn.commit()
    conn.close()
    print("\nMigration completed successfully!")

if __name__ == '__main__':
    migrate()


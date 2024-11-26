import sqlite3

conn = sqlite3.connect("facial_data.db")
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS patient_facial_data (
    patient_id TEXT PRIMARY KEY,
    landmarks BLOB,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()

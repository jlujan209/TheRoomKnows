import sqlite3

conn = sqlite3.connect("patients.db")
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS patient_data (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_first_name TEXT NOT NULL,
    patient_last_name TEXT NOT NULL,
    patient_age INTEGER,
    last_visit TEXT
)
''')

conn.commit()

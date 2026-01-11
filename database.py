import sqlite3  # Assuming you're using SQLite, adjust if using another DB

# Function to initialize the database (if not already created)
def initialize_database():
    conn = sqlite3.connect('your_database.db')  # Change to your DB path
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        detection_count INTEGER,
                        image_blob BLOB)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE,
                        password TEXT)''')  # Assuming a simple users table for login
    conn.commit()
    conn.close()

# Function to save a detection (example, adjust if needed)
def save_detection(timestamp, detection_count, image_blob):
    conn = sqlite3.connect('your_database.db')  # Change to your DB path
    cursor = conn.cursor()
    cursor.execute("INSERT INTO detections (timestamp, detection_count, image_blob) VALUES (?, ?, ?)",
                   (timestamp, detection_count, image_blob))
    conn.commit()
    conn.close()

# Function to fetch detections from the database
def fetch_detections():
    conn = sqlite3.connect('your_database.db')  # Change to your DB path
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections")
    detections = cursor.fetchall()
    conn.close()
    return detections

# Function to clear all detections from the database
def clear_detections():
    conn = sqlite3.connect('your_database.db')  # Change to your DB path
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")  # This will delete all rows in the detections table
    conn.commit()
    conn.close()

# Function to add a user to the users table (for login)
def add_user(username, password):
    conn = sqlite3.connect('your_database.db')  # Change to your DB path
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

# Function to verify user credentials (for login)
def verify_user(username, password):
    conn = sqlite3.connect('your_database.db')  # Change to your DB path
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

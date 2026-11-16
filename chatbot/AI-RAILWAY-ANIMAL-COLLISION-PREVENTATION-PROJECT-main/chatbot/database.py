"""
database.py - SQLite Database Manager for Railway Collision & Animal Detection Safety System.
Stores and retrieves incident queries, AI responses, alert severities, and Kavach actions.
"""

import sqlite3
import datetime
import os
import pandas as pd
from typing import List, Dict, Any, Optional

DB_PATH = "railway_logs.db"

def get_connection():
    """Establish connection to SQLite database with Row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if it doesn't already exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS railway_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                language TEXT DEFAULT 'English',
                hazard_detected INTEGER DEFAULT 0,
                severity TEXT DEFAULT 'NORMAL',
                obstacle_type TEXT DEFAULT 'None',
                kavach_action TEXT DEFAULT 'Normal Supervision',
                audio_file TEXT,
                rag_sources_used INTEGER DEFAULT 0
            )
        """)
        # Create indices for quick filtering and dashboard analytics
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON railway_logs(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hazard ON railway_logs(hazard_detected)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_severity ON railway_logs(severity)")
        conn.commit()

def insert_log(
    query: str,
    response: str,
    language: str = "English",
    hazard_detected: bool = False,
    severity: str = "NORMAL",
    obstacle_type: str = "None",
    kavach_action: str = "Normal Supervision",
    audio_file: Optional[str] = None,
    rag_sources_used: int = 0
) -> int:
    """Inserts a new telemetry log entry and returns its assigned ID."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO railway_logs (
                timestamp, query, response, language, hazard_detected,
                severity, obstacle_type, kavach_action, audio_file, rag_sources_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            query,
            response,
            language,
            1 if hazard_detected else 0,
            severity,
            obstacle_type,
            kavach_action,
            audio_file or "",
            rag_sources_used
        ))
        conn.commit()
        return cursor.lastrowid

def get_all_logs(limit: int = 100, hazard_only: bool = False, search: str = "") -> List[Dict[str, Any]]:
    """Retrieves logs ordered by most recent first, with optional filtering."""
    init_db()
    query = "SELECT * FROM railway_logs WHERE 1=1"
    params = []

    if hazard_only:
        query += " AND hazard_detected = 1"
    
    if search:
        query += " AND (query LIKE ? OR response LIKE ? OR obstacle_type LIKE ?)"
        term = f"%{search}%"
        params.extend([term, term, term])

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_log_stats() -> Dict[str, Any]:
    """Computes telemetry statistics for dashboard widgets."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM railway_logs")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM railway_logs WHERE hazard_detected = 1")
        hazards = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM railway_logs WHERE severity = 'CRITICAL'")
        critical = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM railway_logs WHERE language = 'Tamil'")
        tamil_queries = cursor.fetchone()[0]

        cursor.execute("SELECT obstacle_type, COUNT(*) as cnt FROM railway_logs WHERE obstacle_type != 'None' GROUP BY obstacle_type ORDER BY cnt DESC LIMIT 3")
        top_obstacles = [f"{row[0]} ({row[1]})" for row in cursor.fetchall()]

        return {
            "total_queries": total,
            "total_hazards": hazards,
            "critical_alerts": critical,
            "tamil_queries": tamil_queries,
            "top_obstacles": ", ".join(top_obstacles) if top_obstacles else "None"
        }

def export_logs_dataframe() -> pd.DataFrame:
    """Exports all logs to a pandas DataFrame for Streamlit UI rendering and CSV export."""
    init_db()
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM railway_logs ORDER BY id DESC", conn)
        return df

def clear_logs() -> bool:
    """Clears all records for demonstration reset."""
    with get_connection() as conn:
        conn.cursor().execute("DELETE FROM railway_logs")
        conn.commit()
    return True

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at", DB_PATH)


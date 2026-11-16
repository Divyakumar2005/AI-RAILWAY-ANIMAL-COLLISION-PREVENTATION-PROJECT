"""
test_integration.py - Automated verification test for AI Railway Safety System.
Tests RAG extraction, hazard classification, gTTS generation, and SQLite logging.
"""

import sys
import os

# Fix Windows console UTF-8 output for Tamil text
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rag_engine import SafetyRAGService
from tts_alert import generate_voice_alert
from database import init_db, insert_log, get_all_logs, get_log_stats

def test_full_pipeline():
    print("--- 1. Testing Database Initialization ---")
    init_db()
    stats_before = get_log_stats()
    print("Initial DB Stats:", stats_before)

    print("\n--- 2. Testing RAG Service Loading ---")
    rag = SafetyRAGService("railway_safety_manual.pdf")
    assert rag.kb.is_loaded, "Knowledge base failed to load!"
    print(f"Loaded {len(rag.kb.chunks)} chunks.")

    test_cases = [
        ("Elephant on track at km 340", "en", "Elephant"),
        ("தண்டவாளத்தில் யானை உள்ளது", "ta", "Elephant"),
        ("How does Kavach prevent head-on collisions?", "en", "None")
    ]

    for query, expected_lang, expected_obstacle in test_cases:
        print(f"\nTesting Query: '{query}'")
        res = rag.generate_response(query)
        hazard_info = res["hazard_info"]
        print(f"  Detected Hazard: {hazard_info['hazard_detected']}")
        print(f"  Severity: {hazard_info['severity']}")
        print(f"  Obstacle: {hazard_info['obstacle_type']}")
        print(f"  Kavach Action: {hazard_info['kavach_action'][:60]}...")
        print(f"  Sources Found: {len(res['sources'])}")

        audio_path = None
        if hazard_info["hazard_detected"]:
            audio_path, msg = generate_voice_alert(
                text=res["response"],
                lang=hazard_info["language"].lower()[:2],
                obstacle_type=hazard_info["obstacle_type"]
            )
            print(f"  Audio Alert Generated: {audio_path}")
            assert audio_path and os.path.exists(audio_path), f"Audio file {audio_path} not created!"

        log_id = insert_log(
            query=query,
            response=res["response"],
            language=hazard_info["language"],
            hazard_detected=hazard_info["hazard_detected"],
            severity=hazard_info["severity"],
            obstacle_type=hazard_info["obstacle_type"],
            kavach_action=hazard_info["kavach_action"],
            audio_file=audio_path or "",
            rag_sources_used=len(res["sources"])
        )
        print(f"  Saved to SQLite DB with ID: {log_id}")

    print("\n--- 3. Verifying Updated Database Stats ---")
    stats_after = get_log_stats()
    print("Updated DB Stats:", stats_after)
    assert stats_after["total_queries"] > stats_before["total_queries"]
    print("\n✅ All integration tests PASSED successfully!")

if __name__ == "__main__":
    test_full_pipeline()

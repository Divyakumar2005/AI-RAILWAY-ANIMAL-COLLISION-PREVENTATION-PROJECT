"""
tts_alert.py - Automated Text-to-Speech (gTTS) Alert Engine.
Generates bilingual emergency voice alerts (English & Tamil) for detected hazards
and formats them for browser playback and dashboard audio integration.
"""

import os
import base64
import time
import re
from typing import Optional, Tuple
from gtts import gTTS

ALERTS_DIR = "alerts"

def ensure_alert_dir():
    """Ensures the directory for generated audio alerts exists."""
    os.makedirs(ALERTS_DIR, exist_ok=True)

def detect_language_simple(text: str) -> str:
    """
    Detects if the input text contains Tamil script characters.
    Unicode range for Tamil: \u0B80-\u0BFF
    """
    if re.search(r'[\u0B80-\u0BFF]', text):
        return "ta"
    return "en"

def get_standard_emergency_announcement(obstacle: str, lang: str = "en") -> str:
    """
    Provides crisp, high-clarity emergency voice announcements
    tailored for railway locomotive cab broadcast.
    """
    obstacle_lower = obstacle.lower()
    
    if lang == "ta":
        if "elephant" in obstacle_lower or "யானை" in obstacle_lower:
            return "அவசர எச்சரிக்கை! காட்டுப்பகுதி தண்டவாளத்தில் யானை தென்பட்டுள்ளது. கவச் அவசரகால பிரேக் உடனடியாக இயக்கப்படுகிறது!"
        elif "cow" in obstacle_lower or "cattle" in obstacle_lower or "மாடு" in obstacle_lower:
            return "எச்சரிக்கை! தண்டவாளத்தில் கால்நடைகள் தென்பட்டுள்ளன. ரயிலின் வேகத்தை 20 கிலோமீட்டராக குறைக்கவும்."
        elif "boulder" in obstacle_lower or "rock" in obstacle_lower or "பாறை" in obstacle_lower:
            return "அபாயம்! தண்டவாளத்தில் பாறை அல்லது நிலச்சரிவு ஏற்பட்டுள்ளது. அவசரகால பிரேக் பயன்படுத்தவும்."
        else:
            return "அபாயம்! தண்டவாளத்தில் தடை கண்டறியப்பட்டுள்ளது. கவச் பாதுகாப்பு அமைப்பு செயல்படுகிறது. ரயிலை உடனடியாக நிறுத்தவும்!"
    else:
        if "elephant" in obstacle_lower:
            return "Critical Emergency! Elephant herd detected on track ahead. Kavach emergency brake engaged immediately!"
        elif "cow" in obstacle_lower or "cattle" in obstacle_lower:
            return "Warning! Livestock detected on track ahead. Automatic speed restriction engaged. Reduce speed to 20 kilometers per hour."
        elif "boulder" in obstacle_lower or "rock" in obstacle_lower or "landslide" in obstacle_lower:
            return "Hazard Alert! Track obstruction detected. Emergency brake activated. Bring train to absolute halt."
        else:
            return "Danger! Obstacle detected on railway track. Kavach Automatic Train Protection initiated. Halt train immediately!"

def generate_voice_alert(
    text: str,
    lang: Optional[str] = None,
    obstacle_type: str = "Obstacle",
    slow: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """
    Synthesizes speech using Google Text-to-Speech (gTTS).
    
    Args:
        text: Custom alert text or explanation.
        lang: 'en' for English, 'ta' for Tamil. Auto-detected if None.
        obstacle_type: Type of obstacle for standard railway protocol broadcast.
        slow: Whether to generate speech at a slower pace.

    Returns:
        Tuple of (audio_file_path, announcement_text) or (None, None) on failure.
    """
    ensure_alert_dir()

    # Determine language
    if not lang:
        lang = detect_language_simple(text)

    # If the text is very long or technical, generate the crisp standard emergency announcement
    if len(text) > 180 or not text.strip():
        announcement = get_standard_emergency_announcement(obstacle_type, lang)
    else:
        # Sanitize for TTS
        clean_text = re.sub(r'[*_#`\[\]]', '', text)
        announcement = clean_text[:200]

    filename = f"alert_{lang}_{int(time.time() * 1000)}.mp3"
    filepath = os.path.join(ALERTS_DIR, filename)

    try:
        tts = gTTS(text=announcement, lang=lang, slow=slow)
        tts.save(filepath)
        return filepath, announcement
    except Exception as e:
        # Graceful fallback: Network issue or service rate limit
        print(f"[TTS Error] Could not generate audio alert: {e}")
        return None, announcement

def get_audio_html(filepath: str, autoplay: bool = True) -> str:
    """
    Reads an audio file and converts it into an HTML5 audio tag with base64 encoding.
    Ensures seamless autoplay and cross-browser playback inside Streamlit.
    """
    if not filepath or not os.path.exists(filepath):
        return ""
    
    try:
        with open(filepath, "rb") as f:
            audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
        autoplay_attr = "autoplay" if autoplay else ""
        
        return f"""
        <div style="background: rgba(220, 38, 38, 0.1); border-left: 4px solid #ef4444; padding: 10px 14px; border-radius: 6px; margin: 10px 0;">
            <p style="margin: 0 0 6px 0; font-weight: 600; color: #b91c1c; font-size: 0.9rem;">
                📢 Automated Audio Voice Broadcast (Kavach Cab Alert)
            </p>
            <audio controls {autoplay_attr} style="width: 100%; height: 36px;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        </div>
        """
    except Exception as e:
        print(f"[Audio HTML Error] {e}")
        return ""

if __name__ == "__main__":
    fp, msg = generate_voice_alert("Elephant detected on track", lang="en", obstacle_type="Elephant")
    print(f"Generated test alert at: {fp} with message: {msg}")


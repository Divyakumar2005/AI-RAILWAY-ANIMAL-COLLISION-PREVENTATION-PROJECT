"""
app.py - AI Railway Collision & Animal Detection Safety System.
Production-ready Streamlit Web Application integrating Kavach ATP protocols,
PDF RAG via pypdf, SQLite telemetry logging, gTTS automated voice alerts,
and Google Gemini API with bilingual English/Tamil intelligence.
"""

import sys
import os

# If executed directly via `python app.py`, auto-forward into Streamlit CLI runner
from streamlit.runtime import exists
if not exists():
    from streamlit.web import cli as stcli
    port = os.getenv("PORT", "8501")
    print(f"[Launcher] Starting Streamlit Command Center on http://localhost:{port}...")
    sys.argv = ["streamlit", "run", os.path.abspath(__file__), f"--server.port={port}"]
    sys.exit(stcli.main())

import time
import datetime
import pandas as pd
import streamlit as st

# Local modules
from database import init_db, insert_log, get_all_logs, get_log_stats, export_logs_dataframe, clear_logs
from tts_alert import generate_voice_alert, get_audio_html
from rag_engine import SafetyRAGService, is_tamil_text
from vision_detector import analyze_track_image

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Railway Collision & Animal Detection System",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Railway Command Center CSS
st.markdown("""
<style>
    /* Main background and font styling */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1527 0%, #080c16 90%);
        color: #f1f5f9;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Railway Header Bar */
    .railway-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .railway-title {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .railway-subtitle {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 4px 0 0 0;
    }
    
    /* Danger Indicator Pulse Animations */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 16px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    @keyframes pulse-orange {
        0% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.6); }
        70% { box-shadow: 0 0 0 12px rgba(249, 115, 22, 0); }
        100% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0); }
    }
    
    .status-card-critical {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 16px 20px;
        color: #ffffff;
        animation: pulse-red 1.6s infinite;
        margin-bottom: 16px;
    }
    .status-card-high {
        background: linear-gradient(135deg, #431407 0%, #7c2d12 100%);
        border: 2px solid #f97316;
        border-radius: 10px;
        padding: 16px 20px;
        color: #ffffff;
        animation: pulse-orange 2s infinite;
        margin-bottom: 16px;
    }
    .status-card-medium {
        background: linear-gradient(135deg, #422006 0%, #713f12 100%);
        border: 2px solid #eab308;
        border-radius: 10px;
        padding: 16px 20px;
        color: #ffffff;
        margin-bottom: 16px;
    }
    .status-card-normal {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 16px 20px;
        color: #ffffff;
        margin-bottom: 16px;
    }
    
    /* Telemetry KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }
    .kpi-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #38bdf8;
    }

    /* RAG & AI Response Panel */
    .response-card {
        background: #131d31;
        border: 1px solid #293854;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
    }

    /* Citation badge */
    .citation-badge {
        background: #1e293b;
        border: 1px solid #475569;
        color: #cbd5e1;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        display: inline-block;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Application Initialization & Session State
# ---------------------------------------------------------
init_db()

# Cached RAG Service to avoid re-reading PDF on every rerun
@st.cache_resource(show_spinner="Loading Railway Safety Manual & RAG Vector Index...")
def get_rag_service():
    return SafetyRAGService("railway_safety_manual.pdf")

rag_service = get_rag_service()

# Initialize session state variables
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "latest_audio_path" not in st.session_state:
    st.session_state.latest_audio_path = None
if "active_vision_image" not in st.session_state:
    st.session_state.active_vision_image = None
if "active_vision_filename" not in st.session_state:
    st.session_state.active_vision_filename = ""
if "current_hazard_status" not in st.session_state:
    st.session_state.current_hazard_status = {
        "severity": "NORMAL",
        "obstacle_type": "None",
        "kavach_action": "SIL-4 Normal Supervision",
        "speed_directive": "Sectional Speed (110 km/h)",
        "status_label": "ALL CLEAR / NORMAL",
        "color": "#10b981",
        "hazard_detected": False
    }

# ---------------------------------------------------------
# Sidebar: System Settings, API Key & SQLite DB Logs
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚆 Indian Railways")
    st.title("Control Center")
    st.markdown("---")

    # API Key Configuration
    st.subheader("🔑 Google Gemini API")
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "API Key (Optional / Auto-Fallback)",
        value=env_key,
        type="password",
        help="Paste your Google Gemini API Key. If empty, the app runs on its intelligent local RAG Safety Rulebase!"
    )
    
    # Show API key status
    if api_key_input and api_key_input.strip():
        st.success("✅ API Key entered — Gemini AI mode active")
    else:
        st.info("ℹ️ No API Key — Running in Offline RAG mode")
    
    selected_model = st.selectbox(
        "LLM Model Architecture",
        [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        index=0,
        help="Select which Gemini model to query. Flash = fast & free-tier friendly. Pro = deeper reasoning."
    )

    st.markdown("---")
    
    # PDF Manual Status
    st.subheader("📄 RAG Knowledge Base")
    if rag_service.kb.is_loaded:
        st.success(f"✅ Manual Loaded: `{len(rag_service.kb.chunks)}` chunks indexed from `railway_safety_manual.pdf`")
    else:
        st.error("❌ `railway_safety_manual.pdf` missing. Run generate_manual.py!")

    st.markdown("---")

    # SQLite Database Inspector
    st.subheader("🗄️ SQLite Database Logs")
    stats = get_log_stats()
    
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Total Logs", stats["total_queries"])
    col_s2.metric("Hazards Flagged", stats["total_hazards"])
    
    col_s3, col_s4 = st.columns(2)
    col_s3.metric("Critical Alerts", stats["critical_alerts"])
    col_s4.metric("Tamil Queries", stats["tamil_queries"])
    
    st.caption(f"**Top Obstacles Logged:** {stats['top_obstacles']}")

    # ── Live Analytics Charts ──────────────────────────────────
    if stats["total_queries"] > 0:
        df_all = export_logs_dataframe()
        with st.expander("📊 Live Telemetry Analytics", expanded=False):
            sev_counts = df_all["severity"].value_counts().reindex(
                ["CRITICAL", "HIGH", "MEDIUM", "NORMAL"], fill_value=0
            )
            st.markdown("**Alert Severity Breakdown:**")
            st.bar_chart(sev_counts)

            obs_counts = df_all[df_all["obstacle_type"] != "None"]["obstacle_type"].value_counts()
            if not obs_counts.empty:
                st.markdown("**Detected Obstacle Frequency:**")
                st.bar_chart(obs_counts)

            lang_counts = df_all["language"].value_counts()
            st.markdown("**Query Language Split:**")
            st.bar_chart(lang_counts)

    show_db_logs = st.checkbox("🔍 Inspect `railway_logs.db`", value=False)
    if show_db_logs:
        st.markdown("#### Telemetry Log Audit")
        hazard_filter = st.checkbox("Show Hazards Only", value=False)
        search_term = st.text_input("Filter logs by text:", value="")
        
        logs = get_all_logs(limit=25, hazard_only=hazard_filter, search=search_term)
        if logs:
            df_display = pd.DataFrame(logs)[["id", "timestamp", "query", "severity", "obstacle_type", "kavach_action", "language"]]
            df_display = df_display.copy()
            df_display["query"] = df_display["query"].str[:55] + "..."
            st.dataframe(df_display, use_container_width=True, height=220)
            
            # Export CSV button
            csv_data = export_logs_dataframe().to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Full Logs (CSV)",
                data=csv_data,
                file_name=f"railway_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No logs match the criteria.")

        if st.button("🗑️ Clear Database Records", type="secondary"):
            clear_logs()
            st.session_state.latest_result = None
            st.session_state.latest_audio_path = None
            st.session_state.current_hazard_status = {
                "severity": "NORMAL", "obstacle_type": "None",
                "kavach_action": "SIL-4 Normal Supervision",
                "speed_directive": "Sectional Speed (110 km/h)",
                "status_label": "ALL CLEAR / NORMAL",
                "color": "#10b981", "hazard_detected": False
            }
            st.toast("Logs database cleared for fresh testing!", icon="🧹")
            st.rerun()

    st.markdown("---")
    st.caption("College Capstone Project: AI Railway Collision & Animal Detection Safety System. Powered by Kavach SIL-4, Streamlit, SQLite, gTTS & Gemini.")

# ---------------------------------------------------------
# Main UI: Railway Header & Telemetry HUD
# ---------------------------------------------------------
st.markdown("""
<div class="railway-header">
    <div>
        <h1 class="railway-title">AI Railway Collision & Animal Detection Safety System</h1>
        <p class="railway-subtitle">Indian Railways SIL-4 Telemetry Operations | Kavach Automatic Train Protection & Wildlife Early Warning</p>
    </div>
    <div style="text-align: right;">
        <span style="background: #1e3a8a; border: 1px solid #3b82f6; color: #bfdbfe; font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 20px;">
            📡 KAVACH LINK: ACTIVE (406-470 MHz)
        </span>
        <div style="color: #64748b; font-size: 0.75rem; margin-top: 4px;">
            Local Time: <b>%s</b>
        </div>
    </div>
</div>
""" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), unsafe_allow_html=True)

# ---------------------------------------------------------
# Dynamic Visual Danger Indicator
# ---------------------------------------------------------
curr = st.session_state.current_hazard_status
severity_val = curr.get("severity", "NORMAL")

if severity_val == "CRITICAL":
    card_class = "status-card-critical"
    alert_icon = "🚨"
elif severity_val == "HIGH":
    card_class = "status-card-high"
    alert_icon = "⚠️"
elif severity_val == "MEDIUM":
    card_class = "status-card-medium"
    alert_icon = "⚡"
else:
    card_class = "status-card-normal"
    alert_icon = "🛡️"

st.markdown(f"""
<div class="{card_class}">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">
                LIVE VISUAL DANGER INDICATOR & TRACK SAFETY STATUS
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; margin: 4px 0;">
                {alert_icon} {curr['status_label']}
            </div>
            <div style="font-size: 0.95rem; margin-top: 4px; opacity: 0.95;">
                <b>Detected Target:</b> <span style="background: rgba(0,0,0,0.3); padding: 2px 8px; border-radius: 4px;">{curr['obstacle_type']}</span>
                &nbsp;|&nbsp; 
                <b>Braking Intervention:</b> {curr['kavach_action']}
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.75rem; text-transform: uppercase; opacity: 0.8;">Target Speed Envelope</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #f8fafc;">{curr['speed_directive']}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Audio Voice Alert Widget (plays when hazard is triggered)
if st.session_state.latest_audio_path and os.path.exists(st.session_state.latest_audio_path):
    st.markdown(get_audio_html(st.session_state.latest_audio_path, autoplay=True), unsafe_allow_html=True)

# ---------------------------------------------------------
# Telemetry Sensors KPI Bar
# ---------------------------------------------------------
# Dynamic Speed & Sensor readings based on current hazard status
if severity_val == "CRITICAL":
    disp_speed = "0 km/h (EB Applied)"
    disp_signal = "RED (Stop / Incursion)"
    disp_braking = "EB ENGAGED (-0.85 m/s²)"
    disp_zone = "Walayar Elephant Corridor (Km 512)"
elif severity_val == "HIGH":
    disp_speed = "20 km/h (Caution Order)"
    disp_signal = "DOUBLE YELLOW (Restricted)"
    disp_braking = "SB MODULATING"
    disp_zone = "Palakkad - Madukkarai Sec"
elif severity_val == "MEDIUM":
    disp_speed = "45 km/h"
    disp_signal = "YELLOW (Attention)"
    disp_braking = "VIGILANCE ARMED"
    disp_zone = "Northeast Frontier Corridor"
else:
    disp_speed = "110 km/h"
    disp_signal = "GREEN (Proceed Clear)"
    disp_braking = "STANDBY SUPERVISION"
    disp_zone = "Sector 4 Mainline"

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-box">
        <div class="kpi-label">Locomotive Speed</div>
        <div class="kpi-val" style="color: {'#ef4444' if severity_val=='CRITICAL' else '#38bdf8'};">{disp_speed}</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-label">Track Signal Aspect</div>
        <div class="kpi-val" style="color: {'#ef4444' if severity_val=='CRITICAL' else ('#f97316' if severity_val=='HIGH' else '#10b981')};">{disp_signal}</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-label">Kavach Brake State</div>
        <div class="kpi-val" style="color: {'#ef4444' if severity_val=='CRITICAL' else '#cbd5e1'};">{disp_braking}</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-label">Monitored Corridor</div>
        <div class="kpi-val" style="font-size: 0.95rem; color: #f1f5f9; padding-top: 4px;">{disp_zone}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Interactive Incident Reporter & Simulation Hub (Dual Tab View)
# ---------------------------------------------------------
tab_nlp, tab_vision = st.tabs([
    "📝 Natural Language Incident Reporter (RAG Q&A)",
    "📷 Track Surveillance Camera & Vision AI (Animal Detection)"
])

query_to_process = None
run_vision_analysis = False
vision_img_to_analyze = None
vision_filename = ""
vision_language = "English"

with tab_nlp:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🚨 Safety Query & Hazard Incident Reporter")
        st.markdown("Enter queries or report live track obstacles in **English** or **Tamil (தமிழ்)**:")

        # Quick Simulation Preset Buttons for College Demo
        st.markdown("**⚡ Quick Test Presets (Click to Simulate Scenarios):**")
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        
        if sim_col1.button("🐘 Elephant on Track (EN)", use_container_width=True):
            query_to_process = "Critical alert: Elephant detected crossing track near kilometer 512 in forest corridor. What is the Kavach protocol?"
        if sim_col2.button("🐘 தண்டவாளத்தில் யானை (TA)", use_container_width=True):
            query_to_process = "அவசர எச்சரிக்கை! தண்டவாளத்தில் யானை கூட்டம் நிற்கிறது. கவச் என்ன நடவடிக்கை எடுக்கும்?"
        if sim_col3.button("⚠️ Boulder / Landslide", use_container_width=True):
            query_to_process = "Hazard report: Heavy boulder and landslide fallen onto the tracks ahead after heavy rain. Apply emergency brake."
        
        sim_col4, sim_col5, sim_col6 = st.columns(3)
        if sim_col4.button("🐄 Cattle Herd / மாடு", use_container_width=True):
            query_to_process = "Cattle strayed onto the tracks in agricultural crossing zone. Sound horn and restrict speed."
        if sim_col5.button("🚦 SPAD (Signal Passed at Danger)", use_container_width=True):
            query_to_process = "Locomotive passed signal at danger (RED aspect) approaching junction. How does Kavach prevent collision?"
        if sim_col6.button("📖 Kavach Architecture Q&A", use_container_width=True):
            query_to_process = "Explain the SIL-4 architecture of Kavach ATP, RFID balise tags, and the Plan Bee acoustic system."

        # Input Form
        with st.form("incident_query_form", clear_on_submit=False):
            user_input_text = st.text_area(
                "Incident Description / Safety Query:",
                value=query_to_process if query_to_process else "",
                placeholder="E.g., Elephant detected on track ahead / தண்டவாளத்தில் யானை நிற்கிறது / What is the braking curve of Kavach?",
                height=100
            )
            
            c_sub1, c_sub2 = st.columns([3, 1])
            with c_sub1:
                st.caption("Supports bilingual natural language queries (English and Tamil). Automatically triggers voice broadcast for hazards.")
            with c_sub2:
                submit_btn = st.form_submit_button("🚀 Submit & Analyze", use_container_width=True)

        if submit_btn and user_input_text.strip():
            query_to_process = user_input_text.strip()

    with col_right:
        st.subheader("📡 Live Track Telemetry & Protocols")
        st.markdown("""
        **Core System Components in Action:**
        - **Intrusion Detection System (IDS):** Distributed Acoustic Sensing (DAS) measuring seismic vibrations (10-25 Hz).
        - **Optical & Thermal Towers:** Placed every 400m; detects animals up to 800m.
        - **Plan Bee Deterrents:** 600m directional acoustic honey-bee buzzer.
        - **Loco-Kavach BIU:** Automated Emergency Brake decelerating at `-0.85 m/s²`.
        - **Radio Mesh:** Peer-to-Peer 406-470 MHz UHF emergency packet propagation.
        """)
        
        with st.expander("ℹ️ How to Test for College Viva / Presentation"):
            st.markdown("""
            1. **Test Elephant Detection:** Click **'🐘 Elephant on Track (EN)'** or **'🐘 தண்டவாளத்தில் யானை (TA)'**. Notice the **flashing RED indicator**, **Emergency Brake action**, **audio voice alert**, and **database logging**.
            2. **Test Bilingual Support:** Type in Tamil (e.g. `தண்டவாளத்தில் மாடு உள்ளது` or `கவச் என்றால் என்ன?`). App will answer in Tamil and pronounce a Tamil cab broadcast!
            3. **Inspect SQLite Logs:** Check the sidebar **'Inspect railway_logs.db'** to view logged records and download the CSV report.
            4. **Offline Resilience:** Even if no Gemini API Key is entered, the app functions 100% reliably using the local PDF RAG rulebase.
            """)

with tab_vision:
    st.subheader("📷 Optical & Thermal Track Surveillance (AI Animal & Obstacle Vision)")
    st.markdown("Identify animals visually on tracks from CCTV camera feeds, thermal snapshots, or uploaded track photos:")

    vis_col_left, vis_col_right = st.columns([3, 2])

    with vis_col_left:
        st.markdown("**⚡ Quick Test Surveillance Feeds (Click to Inspect):**")
        v_col1, v_col2, v_col3 = st.columns(3)
        
        sample_img_choice = None
        if v_col1.button("🐘 Cam 402W: Elephant (Walayar)", use_container_width=True):
            sample_img_choice = "sample_images/demo_elephant_track.jpg"
        if v_col2.button("🐄 Cam 108E: Cattle on Track", use_container_width=True):
            sample_img_choice = "sample_images/demo_cattle_track.jpg"
        if v_col3.button("🟢 Cam 012M: Clear Track Ahead", use_container_width=True):
            sample_img_choice = "sample_images/demo_clear_track.jpg"

        if sample_img_choice:
            st.session_state.active_vision_image = sample_img_choice
            st.session_state.active_vision_filename = os.path.basename(sample_img_choice)

        uploaded_img = st.file_uploader("Or upload track snapshot (JPG, PNG):", type=["jpg", "jpeg", "png"])
        if uploaded_img:
            st.session_state.active_vision_image = uploaded_img
            st.session_state.active_vision_filename = uploaded_img.name

        # Display preview and run button if image selected
        if st.session_state.active_vision_image:
            st.image(st.session_state.active_vision_image, caption=f"Active Camera Feed: {st.session_state.active_vision_filename}", use_container_width=True)
            
            c_v_lang1, c_v_lang2 = st.columns([2, 1])
            with c_v_lang1:
                vision_language = st.radio("Voice Alert Language:", ["English", "Tamil (தமிழ்)"], horizontal=True, key="vision_lang_opt")
            with c_v_lang2:
                if st.button("🚀 Analyze with Vision AI", type="primary", use_container_width=True):
                    run_vision_analysis = True
                    vision_img_to_analyze = st.session_state.active_vision_image
                    vision_filename = st.session_state.active_vision_filename

    with vis_col_right:
        st.subheader("📡 Optical & Thermal AI Towers Specification")
        st.markdown("""
        **Surveillance Infrastructure (RDSO Wildlife Directives):**
        - **Placement:** High-mast thermal/optical towers installed every **400 meters** along notified forest corridors.
        - **Spectrum:** Dual-band Daylight HD + Long-Wave Infrared (LWIR) 8-14 µm thermal sensor.
        - **Detection Range:** Accurate animal classification up to **800 meters** in total darkness, fog, and rain.
        - **Kavach Integration:** Real-time edge compute communicates with Station Kavach via wireless mesh.
        - **Automated Actions:** Triggers loco-cab siren, Emergency Brake (EB) if elephant on track, and 'Plan Bee' deterrent sounders.
        """)

# ---------------------------------------------------------
# Processing: Text Query OR Vision AI
# ---------------------------------------------------------
if query_to_process:
    with st.spinner("Processing safety query, analyzing hazards, and cross-referencing railway manual..."):
        try:
            result = rag_service.generate_response(
                query=query_to_process,
                api_key=api_key_input.strip() if api_key_input else None,
                model_name=selected_model
            )
            st.session_state.latest_result = result
            hazard_meta = result["hazard_info"]
            st.session_state.current_hazard_status = hazard_meta

            audio_path = None
            if hazard_meta["hazard_detected"]:
                # Always generate Tamil voice alert for all hazards
                lang_code = "ta"
                audio_path, announcement_text = generate_voice_alert(
                    text=result["response"],
                    lang=lang_code,
                    obstacle_type=hazard_meta["obstacle_type"]
                )
                st.session_state.latest_audio_path = audio_path
            else:
                st.session_state.latest_audio_path = None

            insert_log(
                query=query_to_process,
                response=result["response"],
                language=hazard_meta["language"],
                hazard_detected=hazard_meta["hazard_detected"],
                severity=hazard_meta["severity"],
                obstacle_type=hazard_meta["obstacle_type"],
                kavach_action=hazard_meta["kavach_action"],
                audio_file=audio_path or "",
                rag_sources_used=len(result["sources"])
            )
            st.rerun()
        except Exception as e:
            st.error(f"Error during incident analysis: {str(e)}")
            st.exception(e)

elif run_vision_analysis and vision_img_to_analyze:
    with st.spinner("Analyzing track surveillance camera feed for animals & obstructions..."):
        try:
            lang_target = "Tamil" if "Tamil" in vision_language else "English"
            vision_result = analyze_track_image(
                image_input=vision_img_to_analyze,
                api_key=api_key_input.strip() if api_key_input else None,
                model_name=selected_model,
                language=lang_target,
                image_filename=vision_filename
            )

            # Map to standard result format
            hazard_info = {
                "hazard_detected": vision_result["hazard_detected"],
                "severity": vision_result["severity"],
                "obstacle_type": vision_result["obstacle_type"],
                "kavach_action": vision_result["kavach_action"],
                "speed_directive": vision_result["speed_directive"],
                "color": vision_result["color"],
                "status_label": vision_result["status_label"],
                "language": vision_result["language"]
            }

            st.session_state.current_hazard_status = hazard_info
            
            # Generate Voice Alert - Always in Tamil
            audio_path = None
            if hazard_info["hazard_detected"]:
                lang_code = "ta"  # Always Tamil voice alert
                audio_path, msg = generate_voice_alert(
                    text=vision_result["report_text"],
                    lang=lang_code,
                    obstacle_type=hazard_info["obstacle_type"]
                )
                st.session_state.latest_audio_path = audio_path
            else:
                st.session_state.latest_audio_path = None

            # Insert into SQLite Database
            query_label = f"[Camera Feed: {vision_filename or 'Snapshot'}] Visual Obstacle Inspection"
            insert_log(
                query=query_label,
                response=vision_result["report_text"],
                language=lang_target,
                hazard_detected=hazard_info["hazard_detected"],
                severity=hazard_info["severity"],
                obstacle_type=hazard_info["obstacle_type"],
                kavach_action=hazard_info["kavach_action"],
                audio_file=audio_path or "",
                rag_sources_used=1
            )

            # Store in session state for rendering
            st.session_state.latest_result = {
                "query": query_label,
                "response": vision_result["report_text"],
                "hazard_info": hazard_info,
                "sources": [{
                    "page": 2,
                    "score": 0.98,
                    "snippet": "Section 4.1: AI Optical & Thermal Early Warning Towers placed every 400m recognize elephants and wildlife up to 800m."
                }],
                "llm_engine": vision_result["engine"],
                "language": lang_target
            }
            st.rerun()

        except Exception as e:
            st.error(f"Error during vision analysis: {str(e)}")
            st.exception(e)

# ---------------------------------------------------------
# Response Display & RAG Grounding Citations
# ---------------------------------------------------------
if st.session_state.latest_result:
    res = st.session_state.latest_result
    
    st.markdown("---")
    st.subheader("📋 Safety System Response & Kavach Directives")
    
    col_res1, col_res2 = st.columns([3, 1])
    with col_res1:
        st.caption(f"Engine: **{res['llm_engine']}** | Language: **{res['language']}** | Obstacle: **{res['hazard_info']['obstacle_type']}**")
    with col_res2:
        st.caption(f"Timestamp: **{datetime.datetime.now().strftime('%H:%M:%S')}**")

    # Render AI / System Response inside a styled container
    st.markdown('<div class="response-card">', unsafe_allow_html=True)
    st.markdown(st.session_state.latest_result['response'])
    st.markdown('</div>', unsafe_allow_html=True)

    # Display RAG Source Citations
    if res.get("sources"):
        st.markdown("#### 📚 Referenced Railway Safety Manual Citations (RAG)")
        cit_cols = st.columns(len(res["sources"]))
        for idx, src in enumerate(res["sources"]):
            with cit_cols[idx]:
                st.markdown(f"""
                <div style="background: #1e293b; border-left: 3px solid #38bdf8; padding: 10px 12px; border-radius: 6px; font-size: 0.8rem;">
                    <b>📄 Page {src['page']}</b> &nbsp; <span style="color:#94a3b8;">(Match Score: {src['score']})</span>
                    <p style="margin: 6px 0 0 0; color: #cbd5e1; font-style: italic;">"{src['snippet']}"</p>
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 10px 0;">
    AI Railway Collision & Animal Detection Safety System &copy; 2026 | Built with Python, Streamlit, SQLite, gTTS, pypdf & Google Gemini
</div>
""", unsafe_allow_html=True)

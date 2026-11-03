# AI Railway Collision & Animal Detection Safety System 🚆🐘

A production-ready, full-stack AI web application designed for railway safety operations, wildlife protection, and collision mitigation. This system models India's indigenous **Kavach SIL-4 Automatic Train Protection (ATP)** architecture, combining real-time track telemetry, **pypdf**-driven Retrieval-Augmented Generation (RAG), automated **gTTS** bilingual voice alerts, and the **Google Gemini API**.

---

## 🌟 Key Capabilities & Requirements Implemented

1. **RAG & Document Q&A (pypdf)**:
   - Ingests and processes the official standard operating procedure document (`railway_safety_manual.pdf`).
   - Indexes Kavach ATP subsystems, RFID sleeper spacing, braking curves, axle counters, and elephant corridor directives with TF-IDF cosine vector matching and source page citations.
2. **Automated Hazard & Animal Detection**:
   - Detects obstacles and wildlife (e.g., elephants, cattle, fallen trees, boulders, SPAD red signal breaches).
   - Dynamically triggers a **pulsing visual danger indicator** (CRITICAL RED, HIGH ORANGE, CAUTION YELLOW, ALL CLEAR GREEN).
   - Generates automated **Google Text-to-Speech (gTTS)** voice broadcasts directly into the locomotive cab simulation.
3. **Full Bilingual Support (English & Tamil / தமிழ்)**:
   - Processes natural language queries in both English and Tamil.
   - Generates contextual safety directives and automated voice alerts in the requested language (e.g., Tamil cab voice: *"அவசர எச்சரிக்கை! காட்டுப்பகுதி தண்டவாளத்தில் யானை தென்பட்டுள்ளது..."*).
4. **Local Database Logging (SQLite - `railway_logs.db`)**:
   - Automatically logs every query, AI directive, timestamp, hazard detection flag, severity level, detected obstacle, Kavach braking action, and audio alert path.
   - Interactive sidebar inspector with search filters, hazard toggles, telemetry KPIs, and one-click **CSV export**.
5. **Zero-Failure College Evaluation Architecture**:
   - Supports **Google Gemini 2.5 Flash / 1.5 Flash** via the official `google-genai` SDK.
   - Includes a robust **Offline Safety Rulebase & RAG Matcher** that functions seamlessly even without an active Gemini API key or during offline college viva presentations.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend & UI** | Streamlit | Responsive Railway Command Center HUD & Telemetry Dashboard |
| **Document Processing (RAG)** | `pypdf` + `scikit-learn` | PDF text extraction, chunking, and TF-IDF cosine retrieval |
| **Large Language Model** | Google Gemini API (`google-genai`) | Contextual reasoning, safety analysis & bilingual synthesis |
| **Vision AI (Multimodal)** | Gemini Vision + Computer Vision | Visual animal species identification on track surveillance feeds |
| **Voice Synthesis (TTS)** | `gTTS` (Google Text-to-Speech) | Automated bilingual voice alerts (English & Tamil) |
| **Database** | SQLite3 (`railway_logs.db`) | Audit trail, incident logging, and analytics |
| **Manual Generator** | ReportLab | Generates authentic `railway_safety_manual.pdf` |

---

## 📂 Project Directory Structure

```text
d:\chatbot\
├── app.py                     # Streamlit frontend & main control center
├── rag_engine.py              # pypdf RAG pipeline, hazard classifier & Gemini service
├── vision_detector.py         # Multimodal AI vision engine for track animal detection
├── tts_alert.py               # gTTS automated bilingual voice alert engine
├── database.py                # SQLite database manager for 'railway_logs.db'
├── generate_manual.py         # Script to generate 'railway_safety_manual.pdf'
├── generate_sample_images.py  # Script generating sample track surveillance snapshots
├── sample_images/             # Sample CCTV snapshots (Elephant, Cattle, Clear track)
├── railway_safety_manual.pdf  # Indian Railways Safety SOP document
├── test_integration.py        # Automated end-to-end integration test suite
├── railway_logs.db            # SQLite database storing all telemetry records
├── requirements.txt           # Python dependencies
├── alerts/                    # Cache directory for generated MP3 voice alerts
└── README.md                  # System manual and viva preparation guide
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your machine.
- Windows, macOS, or Linux.

### 2. Installation
Open your terminal in the project directory (`d:\chatbot`) and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. (Optional) Set your Google Gemini API Key
You can set your Gemini API key in your environment:
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-gemini-api-key-here"

# Linux / macOS
export GEMINI_API_KEY="your-gemini-api-key-here"
```
> **Note:** If you don't set an environment variable, you can simply paste your key into the sidebar in the web interface, or leave it empty to test with the built-in offline RAG safety engine!

### 4. Verify or Regenerate the Safety Manual PDF
The manual is pre-generated in the repository. To re-generate it at any time:
```bash
python generate_manual.py
```

### 5. Launch the Web Application
Start the Streamlit application:
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

---

## 🧪 College Viva / Demonstration Scenarios

Use the **Quick Test Preset** buttons on the dashboard to test key scenarios:

### Scenario 1: Elephant Sighted on Track (English)
- **Input:** *"Critical alert: Elephant detected crossing track near kilometer 512 in forest corridor. What is the Kavach protocol?"*
- **Expected Result:**
  - **Visual Danger Indicator:** Flashes **RED** with `CRITICAL EMERGENCY - HALT TRAIN`.
  - **Kavach Action:** Automatic Emergency Brake (EB) applied (-0.85 m/s²), speed clamped to 0 km/h, remote SOS broadcast within 3 km.
  - **Voice Alert:** Automated audio broadcast: *"Critical Emergency! Elephant herd detected on track ahead..."*
  - **Database:** Record logged with severity `CRITICAL` in `railway_logs.db`.

### Scenario 2: தமிழ் விபத்து எச்சரிக்கை (Tamil Query)
- **Input:** *"அவசர எச்சரிக்கை! தண்டவாளத்தில் யானை கூட்டம் நிற்கிறது. கவச் என்ன நடவடிக்கை எடுக்கும்?"*
- **Expected Result:**
  - Language detected as **Tamil**.
  - Visual Danger Indicator pulses in **RED**.
  - Generates clear Tamil safety instructions with Plan Bee deterrent directives.
  - Automated Tamil audio alert generated via `gTTS(lang='ta')`.

### Scenario 3: SPAD (Signal Passed at Danger) Prevention
- **Input:** *"Locomotive passed signal at danger (RED aspect) approaching junction. How does Kavach prevent collision?"*
- **Expected Result:**
  - Identifies SPAD breach.
  - Returns braking curves and RDSO SIL-4 automatic deceleration specifications.

### Scenario 4: General Technical Knowledge (RAG)
- **Input:** *"What is the spacing between RFID tags according to the manual?"*
- **Expected Result:**
  - Status remains **ALL CLEAR / NORMAL (Green)**.
  - Cites Page 1: RFID tags placed every 1000m on open tracks, 100m approaching signals, and 50m in turnout lines.

---

## 🎓 College Viva Preparation Q&A

**Q1: What is Kavach and what safety level does it achieve?**  
*Answer:* Kavach is India's National Automatic Train Protection (ATP) system developed by the RDSO (Research Designs and Standards Organisation). It operates under SIL-4 (Safety Integrity Level 4) certification, meaning the probability of dangerous failure per hour is less than $10^{-8}$.

**Q2: How does the system prevent animal collisions in forest corridors?**  
*Answer:* It integrates a multi-layered approach: (1) AI optical and thermal camera towers placed every 400m, (2) Distributed Acoustic Sensing (DAS) fiber-optic ground sensors detecting elephant footsteps (10-25 Hz), (3) Nighttime Automated Speed Restrictions (ASR) capping train speeds to 20-30 km/h, and (4) Directional "Plan Bee" acoustic deterrents that emit honey bee buzzing sounds audible up to 600m to clear animals safely from the ballast.

**Q3: How does RAG work in this project?**  
*Answer:* `pypdf` parses `railway_safety_manual.pdf` into semantic text chunks with page metadata. A TF-IDF cosine similarity engine ranks the most relevant chunks against the user's query (with cross-lingual Tamil-to-English keyword expansion). The retrieved context is then injected into the Gemini prompt along with strict safety system directives.

**Q4: Where are logs stored and how can they be exported?**  
*Answer:* All transactions are saved in a local SQLite database (`railway_logs.db`). The Streamlit sidebar provides interactive filtering and a one-click CSV download feature.


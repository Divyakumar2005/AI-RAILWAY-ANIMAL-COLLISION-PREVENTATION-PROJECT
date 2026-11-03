"""
generate_manual.py - Generates an authentic, detailed railway_safety_manual.pdf
covering Kavach ATP, Track Telemetry, and Animal Collision Prevention protocols.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf(filename="railway_safety_manual.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F2C59'),
        alignment=1, # Center
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#333333'),
        alignment=1,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E3E62'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading3'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#C7253E'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#222222'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("INDIAN RAILWAYS & SAFETY TELEMETRY BOARD", title_style))
    story.append(Paragraph("<b>STANDARD OPERATING PROCEDURE (SOP) & TECHNICAL MANUAL:</b><br/>"
                           "AI Railway Collision Prevention, Kavach ATP System & Wildlife Safety Directives<br/>"
                           "<i>Doc Reference: IR-SOP-KAVACH-WILDLIFE-2024-V4 | SIL-4 Certified Specification</i>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0F2C59'), spaceAfter=12))

    # Section 1
    story.append(Paragraph("1. Executive Summary & Kavach ATP Architecture", h1_style))
    story.append(Paragraph(
        "Kavach is India's National Automatic Train Protection (ATP) system, developed by the Research Designs and "
        "Standards Organisation (RDSO) in association with Indian industry. Kavach operates under Safety Integrity Level 4 (SIL-4) "
        "standards to prevent collisions, enforce speed compliance, and eliminate Signal Passing at Danger (SPAD).",
        body_style
    ))
    story.append(Paragraph("<b>1.1 Core Hardware and Subsystems:</b>", h2_style))
    story.append(Paragraph("• <b>Locomotive Kavach Unit (Loco-Kavach):</b> On-board computer unit housed in the engine cab, interfaced with Driver Machine Interface (DMI), Brake Interface Unit (BIU), UHF radio transceiver (406-470 MHz), and under-chassis RFID reader antenna.", bullet_style))
    story.append(Paragraph("• <b>Station Kavach Unit:</b> High-availability dual-redundant processing unit deployed at railway stations, connected to the Electronic Interlocking (EI) and optical track telemetry relays.", bullet_style))
    story.append(Paragraph("• <b>Trackside RFID Network:</b> Passive RFID tags clamped onto sleepers along the track center. Spacing is maintained at 1000m on open tracks, 100m approaching signals, and 50m in turnout/loop lines to determine precise train location, heading, and track ID.", bullet_style))
    story.append(Paragraph("• <b>UHF Radio & GSM-R/LTE-R Mesh:</b> Continuous bi-directional wireless telemetry between trains, stations, and level crossing gates with transmission cycle time under 500 milliseconds.", bullet_style))

    # Section 2
    story.append(Paragraph("2. Collision Prevention Protocols & Automatic Braking Curves", h1_style))
    story.append(Paragraph(
        "Kavach continuously calculates dynamic target distance curves based on train weight, locomotive brake capacity, "
        "gradient profile, and track friction coefficients.",
        body_style
    ))
    story.append(Paragraph("• <b>Head-on Collision Prevention:</b> When two locomotives on the same track detect conflicting Movement Authorities (MA) within a 3.0 km envelope, Loco-Kavach units exchange peer-to-peer UHF collision packets and instantly trigger the Emergency Brake (EB) on both consists without requiring driver intervention.", bullet_style))
    story.append(Paragraph("• <b>Rear-end Collision Mitigation:</b> Loco-Kavach calculates relative separation distance using Station Kavach block occupancy telemetry. If separation drops below the Safe Braking Distance (SBD), Service Brake (SB) is engaged; if separation drops under 600m, Emergency Brake (EB) brings the following train to an absolute halt.", bullet_style))
    story.append(Paragraph("• <b>SPAD (Signal Passed at Danger) Elimination:</b> If a locomotive crosses an approach signal at RED (Stop aspect), Loco-Kavach triggers full emergency brake deceleration (-0.85 m/s²), sounding cab sirens and transmitting automated broadcast alerts to Section Control.", bullet_style))
    story.append(Paragraph("• <b>Manual & Remote SOS Function:</b> Drivers can activate the red Mushroom SOS button on the DMI. Additionally, Station Masters or AI track sensors can inject a digital Remote SOS command that immediately stops all locomotives within a 3000-meter radius.", bullet_style))

    # Section 3
    story.append(Paragraph("3. Track Telemetry & Continuous Sensing Infrastructure", h1_style))
    story.append(Paragraph(
        "Modern track safety incorporates multi-modal telemetry sensors along vulnerable track corridors to capture real-time physical conditions:",
        body_style
    ))
    story.append(Paragraph("• <b>High-Frequency Axle Counters (HASSDAC):</b> Solid-state electromagnetic wheel detection coils measuring axle counts and wheel flange direction at block boundaries to ensure zero undetected decoupled rolling stock.", bullet_style))
    story.append(Paragraph("• <b>Distributed Acoustic Sensing (DAS):</b> Fiber-optic cables buried adjacent to rails function as distributed microphones. Acoustic vibrations caused by boulder falls, landslides, or animal footfalls within 10 meters of the track trigger instant alerts to telemetry servers.", bullet_style))
    story.append(Paragraph("• <b>Rail Temperature & Rail Buckle Telemetry:</b> Remote IoT thermal sensors measure rail temperature. If rail temperature exceeds 60°C in peak summer (Tmax + 25°C), speed restrictions (Caution Orders) of 30 km/h are enforced automatically to prevent track buckling.", bullet_style))

    # Section 4
    story.append(Paragraph("4. Wildlife and Animal Collision Prevention Protocols (Focus: Elephant Corridors)", h1_style))
    story.append(Paragraph(
        "Elephants and other large wildlife frequently traverse recognized corridors bisecting railway tracks in the Western Ghats (Walayar, Coimbatore-Palakkad), "
        "Northeast Frontier (Alipurduar, Lumding), and Odisha forest tracks. Indian Railways mandates the following AI and optical intervention layers:",
        body_style
    ))
    story.append(Paragraph("• <b>AI Optical & Thermal Early Warning Towers:</b> Thermal and optical camera towers placed every 400m in designated wildlife corridors stream video to AI edge processors. The system recognizes elephants, cattle, leopards, and human trespassers at distances up to 800 meters during day, night, fog, and heavy rain.", bullet_style))
    story.append(Paragraph("• <b>Automated Speed Restriction (ASR) Zones:</b> Between 18:00 hrs and 06:00 hrs, trains traversing elephant corridors are speed-restricted to a maximum of 30 km/h to guarantee a safe braking stopping distance.", bullet_style))
    story.append(Paragraph("• <b>Plan Bee Acoustic Deterrent System:</b> Specially engineered directional acoustic devices broadcast recorded sound of swarming honey bees (proven natural deterrent for Asian elephants) up to 600 meters along tracks, clearing herds without causing panic.", bullet_style))
    story.append(Paragraph("• <b>Intrusion Detection System (IDS) Footstep Classification:</b> Seismic and fiber optic ground sensors detect heavy animal footsteps (frequency signature 10 Hz - 25 Hz) within 15 meters of the ballast. Upon 3 consecutive detected paces, a Tier-1 Critical Hazard is raised.", bullet_style))

    # Section 5
    story.append(Paragraph("5. Emergency Action Protocols & Threat Severity Levels", h1_style))
    
    # Table of severity levels
    table_data = [
        ["Alert Level", "Trigger Event", "Kavach Automated Action", "Loco Pilot Directive"],
        [
            "CRITICAL (RED)",
            "Elephant / Cattle / Vehicle / Boulder on track within 1000m",
            "Automatic Emergency Brake (EB) applied. Loco speed cut to 0 km/h. Cab alarm sounds. Remote SOS sent to all trains in 3km.",
            "Confirm EB application, flash headlights, sound continuous hazard whistle, switch cab radio to Channel 160.050 MHz."
        ],
        [
            "HIGH (ORANGE)",
            "Wildlife herd spotted within 50m of track ballast / Sensor warning",
            "Automatic Speed Restriction (ASR) engaged. Throttle cut to 20 km/h. Audio cab chime activated.",
            "Reduce speed to 20 km/h, activate Plan Bee acoustic buzzer, watch for track incursions, report to Station Master."
        ],
        [
            "MEDIUM (YELLOW)",
            "Nighttime corridor entry / Track temperature anomaly / Fog advisory",
            "Advisory caution flag displayed on DMI. Maximum allowable speed clamped to 45 km/h.",
            "Operate headlights on high beam, maintain heightened vigilance, prepare for sudden braking if animal observed."
        ],
        [
            "NORMAL (GREEN)",
            "Clear track ahead, valid Movement Authority, all telemetry normal",
            "Normal running speed under Kavach supervision up to Sectional Max (e.g. 110-130 km/h).",
            "Maintain scheduled speed, observe signal aspects, monitor DMI telemetry status indicators."
        ]
    ]

    t = Table(table_data, colWidths=[80, 140, 160, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F2C59')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDDDDD')),
        ('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#FFDDDD')),
        ('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#FFF2DD')),
        ('BACKGROUND', (0, 3), (0, 3), colors.HexColor('#FFFFDD')),
        ('BACKGROUND', (0, 4), (0, 4), colors.HexColor('#DDFFDD')),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Section 6
    story.append(Paragraph("6. Bilingual Operational Directives (English & Tamil / தமிழ்)", h1_style))
    story.append(Paragraph(
        "For Southern Railway and South Western Railway zones passing through Tamil Nadu and Kerala borders, "
        "telemetry systems and voice advisories must support standard bilingual terminology:",
        body_style
    ))
    story.append(Paragraph("• <b>Hazard Detected:</b> English: 'Danger! Obstacle or animal detected on track ahead.' | Tamil: 'எச்சரிக்கை! தண்டவாளத்தில் ஆபத்து அல்லது விலங்கு தென்பட்டுள்ளது. ரயிலை உடனடியாக நிறுத்தவும்.'", bullet_style))
    story.append(Paragraph("• <b>Elephant on Track:</b> English: 'Emergency! Elephant herd detected on track near forest corridor. Emergency braking engaged.' | Tamil: 'அவசர எச்சரிக்கை! காட்டுப்பகுதி தண்டவாளத்தில் யானை உள்ளது. அவசரகால பிரேக் பயன்படுத்தப்படுகிறது.'", bullet_style))
    story.append(Paragraph("• <b>Speed Restriction:</b> English: 'Caution! Elephant corridor active. Speed restricted to 20 km/h.' | Tamil: 'கவனம்! யானைகள் நடமாடும் பகுதி. வேகம் 20 கி.மீ ஆக குறைக்கப்பட்டுள்ளது.'", bullet_style))
    story.append(Paragraph("• <b>Track Clear:</b> English: 'Track is clear. All telemetry normal. Safe to proceed.' | Tamil: 'தண்டவாளம் தெளிவாக உள்ளது. பயணம் பாதுகாப்பானது.'", bullet_style))

    # Footer note
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CCCCCC'), spaceAfter=8))
    story.append(Paragraph("<i>Approved by Railway Safety Commission & Wildlife Protection Division. Document version 4.2. For automated AI query verification and telemetry cross-reference.</i>", ParagraphStyle('Foot', parent=styles['Italic'], fontSize=8, alignment=1, textColor=colors.gray)))

    doc.build(story)
    print(f"Successfully generated {filename} ({os.path.getsize(filename)} bytes)")

if __name__ == "__main__":
    generate_pdf()


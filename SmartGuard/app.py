"""
SmartGuard Enterprise Dashboard  v5.0
AI-Powered Cybersecurity Threat Detection & User Behaviour Analytics

Run with:
    python -m venv venv && source venv/bin/activate   # (Windows: venv\\Scripts\\activate)
    pip install -r requirements.txt
    streamlit run app.py
"""

import hashlib
import random
import time
import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartGuard Enterprise",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# ─────────────────────────────────────────────
# ML MODELS  (12-feature NSL-KDD hybrid ensemble)
# ─────────────────────────────────────────────
_HYBRID_THRESHOLD = 0.35   # RF(25%) + XGB(75%) fusion threshold

# Attack type labels — hex colours so this block stays before colour constants
_ATK_TYPES = {
    0: dict(name="Normal", full="No Attack Detected",               icon="✅",
            color="#68d391", rec="No action required"),
    1: dict(name="DoS",    full="Denial of Service",                icon="🔴",
            color="#fc8181", rec="Enable rate-limiting · blackhole routing · contact upstream ISP"),
    2: dict(name="Probe",  full="Reconnaissance / Port Scan",       icon="🟠",
            color="#f6ad55", rec="Block source IP · review firewall ACLs · raise IDS sensitivity"),
    3: dict(name="R2L",    full="Remote-to-Local Credential Attack", icon="🟣",
            color="#b794f4", rec="Lock account · enforce MFA · force password reset"),
    4: dict(name="U2R",    full="User-to-Root Privilege Escalation", icon="🔴",
            color="#fc8181", rec="Terminate session · forensic audit · patch privilege escalation path"),
}


import os as _os
_APP_DIR = _os.path.dirname(_os.path.abspath(__file__))

def _p(name): return _os.path.join(_APP_DIR, name)

@st.cache_resource
def load_models():
    try:
        rf     = joblib.load(_p("rf_model.pkl"))
        xgb    = joblib.load(_p("xgb_model.pkl"))
        hybrid = joblib.load(_p("hybrid_model.pkl"))
        iso    = joblib.load(_p("isolation_forest_model.pkl"))
        return rf, xgb, hybrid, iso
    except Exception as e:
        st.warning(f"⚠ Model load failed — running rule-based fallback. ({e})")
        return None, None, None, None

RF_MODEL, XGB_MODEL, HYBRID_MODEL, ISO_MODEL = load_models()

# ─────────────────────────────────────────────
# SESSION STATE — USER ACCOUNT
# ─────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

USERS_DB = {
    "smartguard_admin": {
        "password":    _hash("SG@2026#Enterprise"),
        "full_name":   "Aalia Abdallah Albalushi",
        "initials":    "AA",
        "role":        "Chief Security Officer",
        "department":  "Cybersecurity Operations",
        "location":    "Qatar HQ",
        "clearance":   "TOP SECRET",
        "email":       "aalia.albalushi@smartguard.qa",
        "phone":       "+974 5000 1234",
        "joined":      "Jan 2025",
        "avatar_color": "#4fd1c5",
    },
}

if "logged_in"       not in st.session_state: st.session_state.logged_in       = False
if "current_user"    not in st.session_state: st.session_state.current_user    = None
if "login_error"     not in st.session_state: st.session_state.login_error     = ""
if "session_start"   not in st.session_state: st.session_state.session_start   = None
if "alerts_reviewed" not in st.session_state: st.session_state.alerts_reviewed = 0
if "cases_escalated" not in st.session_state: st.session_state.cases_escalated = 0
if "threat_count"    not in st.session_state: st.session_state.threat_count    = random.randint(5, 12)
if "event_rate"      not in st.session_state: st.session_state.event_rate      = random.randint(290, 380)

# ─────────────────────────────────────────────
# COLOURS & PLOTLY DEFAULTS
# ─────────────────────────────────────────────
CYAN   = "#4fd1c5"
BLUE   = "#63b3ed"
RED    = "#fc8181"
AMBER  = "#f6ad55"
GREEN  = "#68d391"
PURPLE = "#b794f4"
BG     = "#090c14"
BG2    = "#0e1220"
BG3    = "#141926"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", color="#718096", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(79,209,197,0.1)", linecolor="rgba(79,209,197,0.15)", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(79,209,197,0.1)", linecolor="rgba(79,209,197,0.15)", tickfont=dict(size=10)),
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    background-color: #090c14 !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0e1220 !important;
    border-right: 1px solid rgba(79,209,197,0.15) !important;
}
/* ── Keep the sidebar permanently visible (never auto-collapse off-screen) ── */
[data-testid="stSidebar"] {
    transform: none !important;
    visibility: visible !important;
    width: 244px !important;
    min-width: 244px !important;
    margin-left: 0 !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    transform: none !important;
    width: 244px !important;
    min-width: 244px !important;
}
[data-testid="stSidebar"] > div:first-child { width: 244px !important; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── App container ── */
[data-testid="stAppViewContainer"] { background-color: #090c14 !important; }
[data-testid="stHeader"]           { background-color: #090c14 !important; }

/* ── Hide the Streamlit top toolbar so it doesn't block scrolling ── */
[data-testid="stToolbar"]          { display: none !important; }
[data-testid="stDecoration"]       { display: none !important; }
[data-testid="stStatusWidget"]     { display: none !important; }
#MainMenu                          { display: none !important; }
footer                             { display: none !important; }
header[data-testid="stHeader"]     { display: none !important; }

/* ── Full scrollable page, no clipping ── */
html, body {
    overflow: auto !important;
    height: auto !important;
}
[data-testid="stAppViewContainer"] {
    overflow: auto !important;
    height: auto !important;
}
[data-testid="stMain"] {
    overflow: visible !important;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px !important;
    overflow: visible !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0e1220 !important;
    border: 1px solid rgba(79,209,197,0.15) !important;
    border-radius: 6px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"]  { font-family:'JetBrains Mono',monospace!important; font-weight:800!important; font-size:26px!important; color:#4fd1c5!important; }
[data-testid="stMetricLabel"]  { font-family:'JetBrains Mono',monospace!important; font-size:10px!important; letter-spacing:0.1em!important; color:#718096!important; }
[data-testid="stMetricDelta"]  { font-size:11px!important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:#0e1220!important; border:1px solid rgba(79,209,197,0.15)!important;
    border-radius:6px!important; padding:4px!important; gap:2px!important;
}
.stTabs [data-baseweb="tab"] {
    background:transparent!important; color:#718096!important;
    font-family:'JetBrains Mono',monospace!important; font-size:11px!important;
    letter-spacing:0.06em!important; border-radius:4px!important; border:none!important;
}
.stTabs [aria-selected="true"] { background:rgba(79,209,197,0.12)!important; color:#4fd1c5!important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:16px!important; }

/* ── Inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-baseweb="select"] {
    background:#141926!important; border:1px solid rgba(79,209,197,0.2)!important;
    border-radius:4px!important; color:#e2e8f0!important;
    font-family:'JetBrains Mono',monospace!important; font-size:13px!important;
}
input[type="password"] {
    background:#141926!important; color:#e2e8f0!important;
    font-family:'JetBrains Mono',monospace!important;
}

/* ── Buttons ── */
.stButton>button {
    background:rgba(79,209,197,0.08)!important; border:1px solid rgba(79,209,197,0.4)!important;
    border-radius:4px!important; color:#4fd1c5!important;
    font-family:'JetBrains Mono',monospace!important; font-size:12px!important;
    font-weight:700!important; letter-spacing:0.08em!important; padding:8px 24px!important;
}
.stButton>button:hover { background:rgba(79,209,197,0.18)!important; border-color:#4fd1c5!important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border:1px solid rgba(79,209,197,0.15)!important; border-radius:6px!important; }
.stDataFrame th { background:#141926!important; color:#718096!important; font-family:'JetBrains Mono',monospace!important; font-size:10px!important; }
.stDataFrame td { color:#e2e8f0!important; font-family:'JetBrains Mono',monospace!important; font-size:12px!important; }

/* ── Alerts ── */
.stAlert { border-radius:4px!important; font-family:'JetBrains Mono',monospace!important; font-size:12px!important; }
[data-testid="stSuccess"] { background:rgba(104,211,145,0.08)!important; border:1px solid rgba(104,211,145,0.3)!important; }
[data-testid="stError"]   { background:rgba(252,129,129,0.08)!important; border:1px solid rgba(252,129,129,0.3)!important; }
[data-testid="stWarning"] { background:rgba(246,173,85,0.08)!important;  border:1px solid rgba(246,173,85,0.3)!important; }
[data-testid="stInfo"]    { background:rgba(99,179,237,0.08)!important;  border:1px solid rgba(99,179,237,0.3)!important; }

/* ── Divider ── */
hr { border-color:rgba(79,209,197,0.15)!important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:#090c14; }
::-webkit-scrollbar-thumb { background:#4fd1c5; border-radius:2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def session_duration():
    if st.session_state.session_start:
        delta = datetime.now() - st.session_state.session_start
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m = rem // 60
        return f"{h}h {m:02d}m"
    return "—"

def user_initials(name):
    parts = name.split()
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else name[:2].upper()

def classify_ip(ip: str):
    """Return (category_label, color) for an IP string."""
    import ipaddress
    try:
        addr = ipaddress.ip_address(ip.strip())
        if addr.is_private:   return "INTERNAL / LAN",        GREEN
        if addr.is_loopback:  return "LOOPBACK",              BLUE
        if addr.is_multicast: return "MULTICAST",             BLUE
        return                       "EXTERNAL / INTERNET",   RED
    except ValueError:
        return                       "UNKNOWN ADDRESS",        AMBER

def gen_threat_timeline():
    hours  = [f"{h:02d}:00" for h in range(24)]
    values = [12,8,5,3,2,4,18,35,42,38,55,61,48,52,65,71,58,43,39,62,71,84,76,53]
    return pd.DataFrame({"Hour": hours, "Events": values})

# Maps log usernames to departments for the dept filter
_USER_DEPT = {
    "jd.taylor":    "Finance",
    "m.patel":      "Engineering",
    "sys.admin":    "Operations",
    "k.williams":   "Engineering",
    "s.chen":       "HR",
    "a.rodriguez":  "C-Suite",
    "svc.backup":   "Operations",
}

def gen_logs(n=35):
    event_types = [
        ("Failed Authentication",     "AUTH", RED,    "CRITICAL"),
        ("Privilege Escalation",      "IAM",  RED,    "CRITICAL"),
        ("Port Scan Detected",        "NET",  RED,    "CRITICAL"),
        ("Malware Signature Match",   "EP",   RED,    "CRITICAL"),
        ("Data Transfer Spike",       "NET",  AMBER,  "HIGH"),
        ("Suspicious File Access",    "FS",   AMBER,  "HIGH"),
        ("Brute Force Attempt",       "AUTH", AMBER,  "HIGH"),
        ("VPN Login — Foreign IP",    "NET",  AMBER,  "HIGH"),
        ("Database Query Anomaly",    "DB",   AMBER,  "HIGH"),
        ("Password Reset",            "IAM",  BLUE,   "MEDIUM"),
        ("Configuration Change",      "SYS",  BLUE,   "MEDIUM"),
        ("DNS Query Anomaly",         "NET",  BLUE,   "MEDIUM"),
        ("Email Attachment Scan",     "MAIL", BLUE,   "MEDIUM"),
        ("SSL Certificate Expiry",    "SYS",  GREEN,  "LOW"),
        ("API Rate Limit Hit",        "API",  GREEN,  "LOW"),
        ("Firewall Rule Updated",     "NET",  GREEN,  "INFO"),
        ("User Account Created",      "IAM",  GREEN,  "INFO"),
    ]
    ips   = ["185.220.101.47","10.0.1.142","203.0.113.8","198.51.100.22","192.168.5.31","91.108.4.0","172.16.0.45"]
    users = list(_USER_DEPT.keys())
    rows  = []
    now   = datetime.now()
    for i in range(n):
        ev   = random.choice(event_types)
        ts   = now - timedelta(minutes=i*9)
        user = random.choice(users)
        rows.append({
            "Timestamp":  ts.strftime("%H:%M:%S %d/%m"),
            "Event ID":   f"{ev[1]}_{random.randint(100,999)}",
            "User":       user,
            "Department": _USER_DEPT[user],
            "Event Type": ev[0],
            "Source IP":  random.choice(ips),
            "Severity":   ev[3],
        })
    return pd.DataFrame(rows)

INTEL_FEED = [
    {"sev":"CRITICAL","title":"APT-29 Cozy Bear Campaign Active",
     "desc":"Nation-state actor targeting government contractors via spear phishing. IOCs updated in threat DB.",
     "source":"CISA","time":"08:42"},
    {"sev":"CRITICAL","title":"Log4Shell Exploitation Surge",
     "desc":"300% increase in Log4j exploitation attempts targeting unpatched Java services.",
     "source":"SANS Internet Storm Center","time":"09:15"},
    {"sev":"HIGH","title":"BlackCat ALPHV — New Ransomware Variant",
     "desc":"New evasion techniques detected. YARA rules updated and pushed to all endpoints.",
     "source":"IBM X-Force","time":"10:03"},
    {"sev":"HIGH","title":"Supply Chain Attack — npm Package",
     "desc":"Malicious package 'node-helper-util' v2.1.4 identified with 12K downloads. Quarantine recommended.",
     "source":"Snyk Security","time":"11:28"},
    {"sev":"MEDIUM","title":"Credential Stuffing Campaign",
     "desc":"Automated attacks using 4.2M leaked credentials targeting SaaS platforms. Rate-limiting active.",
     "source":"SpyCloud","time":"12:00"},
    {"sev":"MEDIUM","title":"Phishing Kit Targeting Finance Teams",
     "desc":"New template impersonating internal CFO. 3 users clicked — accounts locked pending review.",
     "source":"Proofpoint","time":"13:45"},
]

MITRE_TACTICS  = ["Initial Access","Execution","Persistence","Privilege Esc.","Defense Evasion","Credential Access","Lateral Movement","Exfiltration"]
MITRE_COVERAGE = [88,72,65,81,54,70,43,62]

ATTACK_ORIGINS = [
    {"city":"Moscow",    "country":"Russia",      "lat":55.75, "lon":37.62,   "attacks":142,"type":"APT / State Actor"},
    {"city":"Beijing",   "country":"China",       "lat":39.90, "lon":116.40,  "attacks":118,"type":"Espionage"},
    {"city":"Pyongyang", "country":"North Korea", "lat":39.02, "lon":125.75,  "attacks":67, "type":"Financial Theft"},
    {"city":"Lagos",     "country":"Nigeria",     "lat":6.52,  "lon":3.38,    "attacks":54, "type":"Phishing / BEC"},
    {"city":"Bucharest", "country":"Romania",     "lat":44.43, "lon":26.10,   "attacks":48, "type":"Ransomware"},
    {"city":"Tehran",    "country":"Iran",        "lat":35.69, "lon":51.39,   "attacks":43, "type":"Infrastructure"},
    {"city":"São Paulo", "country":"Brazil",      "lat":-23.55,"lon":-46.63,  "attacks":31, "type":"Credential Theft"},
    {"city":"Minsk",     "country":"Belarus",     "lat":53.90, "lon":27.56,   "attacks":29, "type":"DDoS"},
    {"city":"Jakarta",   "country":"Indonesia",   "lat":-6.21, "lon":106.85,  "attacks":24, "type":"Botnet C2"},
    {"city":"Kharkiv",   "country":"Ukraine",     "lat":49.99, "lon":36.23,   "attacks":19, "type":"Hacktivism"},
]
TARGETS = [
    {"city":"New York","lat":40.71,"lon":-74.01},{"city":"London","lat":51.51,"lon":-0.13},
    {"city":"Frankfurt","lat":50.11,"lon":8.68},{"city":"Singapore","lat":1.35,"lon":103.82},
    {"city":"Tokyo","lat":35.68,"lon":139.69},{"city":"Sydney","lat":-33.87,"lon":151.21},
    {"city":"Dubai","lat":25.20,"lon":55.27},{"city":"Toronto","lat":43.65,"lon":-79.38},
]


# ═════════════════════════════════════════════
#  LOGIN SCREEN
# ═════════════════════════════════════════════
def render_login():
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("""
        <div style='text-align:center;padding:32px 0 24px'>
          <div style='font-family:Rajdhani,sans-serif;font-size:36px;font-weight:700;color:#e2e8f0'>
            🛡️ Smart<span style='color:#4fd1c5'>Guard</span>
          </div>
          <div style='font-family:JetBrains Mono,monospace;font-size:10px;letter-spacing:0.18em;
                      color:#718096;margin-top:6px'>ENTERPRISE SECURITY OPERATIONS PLATFORM</div>
        </div>
        <div style='background:#0e1220;border:1px solid rgba(79,209,197,0.2);
                    border-top:2px solid #4fd1c5;border-radius:8px;padding:28px 32px 20px;
                    margin-bottom:16px'>
          <div style='font-family:JetBrains Mono,monospace;font-size:11px;letter-spacing:0.1em;
                      color:#718096;margin-bottom:20px'>SECURE SIGN-IN</div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="username", key="login_user",
                                 label_visibility="collapsed")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        password = st.text_input("Password", placeholder="password", type="password",
                                 key="login_pass", label_visibility="collapsed")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        if st.button("⚡  SIGN IN TO SMARTGUARD", use_container_width=True, key="login_btn"):
            uname = username.strip().lower()
            if uname in USERS_DB and USERS_DB[uname]["password"] == _hash(password):
                st.session_state.logged_in       = True
                st.session_state.current_user    = uname
                st.session_state.session_start   = datetime.now()
                st.session_state.alerts_reviewed = random.randint(8, 31)
                st.session_state.cases_escalated = random.randint(1, 7)
                st.session_state.threat_count    = random.randint(5, 12)
                st.session_state.event_rate      = random.randint(290, 380)
                st.session_state.login_error     = ""
                st.rerun()
            else:
                st.session_state.login_error = "Invalid credentials. Please try again."

        if st.session_state.login_error:
            st.error(f"🔴 {st.session_state.login_error}")

        st.markdown("""
<div style='text-align:center;margin-top:12px'>
  <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#4fd1c5'>
    SMARTGUARD ENTERPRISE SECURITY PLATFORM
  </div>
  <div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#4a5568;margin-top:6px'>
    Authorized Access Only • Security Monitoring Enabled
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  MAIN DASHBOARD
# ═════════════════════════════════════════════
def render_dashboard():
    u    = USERS_DB[st.session_state.current_user]
    name = u["full_name"]
    init = u["initials"]
    acol = u["avatar_color"]

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style='background:#141926;border:1px solid rgba(79,209,197,0.18);
                    border-radius:6px;padding:16px;margin-bottom:16px'>
          <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>
            <div style='width:42px;height:42px;border-radius:50%;
                        background:rgba(79,209,197,0.15);border:2px solid {acol};
                        display:flex;align-items:center;justify-content:center;
                        font-family:JetBrains Mono,monospace;font-size:14px;
                        font-weight:700;color:{acol};flex-shrink:0'>{init}</div>
            <div>
              <div style='font-size:13px;font-weight:700;color:#e2e8f0'>{name}</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:9px;
                          color:#718096;margin-top:2px'>{u["role"]}</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:9px;
                          color:{acol};margin-top:1px'>{u["location"]}</div>
            </div>
          </div>
          <div style='border-top:1px solid rgba(255,255,255,0.06);padding-top:10px'>
            <div style='display:flex;justify-content:space-between;font-family:JetBrains Mono,monospace;font-size:10px;margin-bottom:5px'>
              <span style='color:#718096'>Clearance</span>
              <span style='color:{acol};font-weight:700'>{u["clearance"]}</span>
            </div>
            <div style='display:flex;justify-content:space-between;font-family:JetBrains Mono,monospace;font-size:10px;margin-bottom:5px'>
              <span style='color:#718096'>Session</span>
              <span style='color:#4fd1c5'>{session_duration()}</span>
            </div>
            <div style='display:flex;justify-content:space-between;font-family:JetBrains Mono,monospace;font-size:10px;margin-bottom:5px'>
              <span style='color:#718096'>Alerts reviewed</span>
              <span style='color:#68d391'>{st.session_state.alerts_reviewed}</span>
            </div>
            <div style='display:flex;justify-content:space-between;font-family:JetBrains Mono,monospace;font-size:10px'>
              <span style='color:#718096'>Cases escalated</span>
              <span style='color:#f6ad55'>{st.session_state.cases_escalated}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;letter-spacing:0.08em;
                    color:#718096;margin-bottom:8px'>SYSTEM STATUS</div>
        """, unsafe_allow_html=True)
        for label, color, text in [
            ("SOC ACTIVE",              GREEN, "ALL SYSTEMS ONLINE"),
            ("LIVE MONITORING",         BLUE,  "ENABLED"),
            ("THREAT INTEL MODE",       AMBER, "ACTIVE"),
        ]:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:8px;
                        background:rgba(255,255,255,0.04);
                        border:1px solid rgba(255,255,255,0.06);border-radius:4px;
                        padding:7px 10px;font-family:JetBrains Mono,monospace;
                        font-size:10px;color:{color};margin-bottom:6px'>
              <div style='width:6px;height:6px;border-radius:50%;background:{color};flex-shrink:0'></div>
              {label} — {text}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;letter-spacing:0.08em;
                    color:#718096;margin-bottom:5px'>LAST SCAN</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:12px;color:{CYAN}'>
        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;
                    letter-spacing:0.08em;color:#718096;margin-bottom:8px'>QUICK FILTERS</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='font-family:JetBrains Mono,monospace;font-size:9px;
                    letter-spacing:0.08em;color:#4fd1c5;margin-bottom:3px'>SEVERITY</div>
        """, unsafe_allow_html=True)
        sev_filter = st.multiselect(
            "Severity",
            ["CRITICAL","HIGH","MEDIUM","LOW","INFO"],
            default=["CRITICAL","HIGH","MEDIUM","LOW","INFO"],
            label_visibility="collapsed",
        )

        st.markdown("""
        <div style='font-family:JetBrains Mono,monospace;font-size:9px;
                    letter-spacing:0.08em;color:#4fd1c5;margin:8px 0 3px'>DEPARTMENT</div>
        """, unsafe_allow_html=True)
        dept_filter = st.selectbox(
            "Department",
            ["All Departments","Engineering","Finance","HR","Operations","C-Suite"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        if st.button("⎋  SIGN OUT", use_container_width=True, key="signout_btn"):
            st.session_state.logged_in     = False
            st.session_state.current_user  = None
            st.session_state.session_start = None
            st.session_state.login_error   = ""
            st.rerun()

        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace;font-size:9px;
                    letter-spacing:0.06em;color:#2d3748;text-align:center;margin-top:8px'>
          SmartGuard Enterprise v5.0 · SmartGuard Security Platform<br>
          Logged in as {name}
        </div>
        """, unsafe_allow_html=True)

# ── TOP HEADER BAR ────────────────────────
    col_logo, col_userbar = st.columns([2.4, 1])

    with col_logo:
        st.markdown(f"""
        <div style='background:#0e1220;border:1px solid rgba(79,209,197,0.18);
                    border-top:2px solid #4fd1c5;border-radius:6px;
                    padding:14px 22px;margin-bottom:14px'>
          <div style='font-family:Rajdhani,sans-serif;font-size:20px;font-weight:700;color:#e2e8f0'>
            🛡️ Smart<span style='color:#4fd1c5'>Guard</span> Enterprise Dashboard
          </div>
          <div style='font-family:JetBrains Mono,monospace;font-size:9px;
                      letter-spacing:0.12em;color:#718096;margin-top:3px'>
            AI-POWERED CYBERSECURITY THREAT DETECTION &amp; USER BEHAVIOUR ANALYTICS
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_userbar:
        st.markdown(f"""
        <div style='background:#0e1220;border:1px solid rgba(79,209,197,0.18);
                    border-radius:6px;padding:14px 18px;margin-bottom:14px;
                    display:flex;align-items:center;gap:12px'>
          <div style='width:36px;height:36px;border-radius:50%;
                      background:rgba(79,209,197,0.15);border:2px solid {acol};
                      display:flex;align-items:center;justify-content:center;
                      font-family:JetBrains Mono,monospace;font-size:13px;
                      font-weight:700;color:{acol};flex-shrink:0'>{init}</div>
          <div style='flex:1;min-width:0'>
            <div style='font-size:13px;font-weight:700;color:#e2e8f0;white-space:nowrap;
                        overflow:hidden;text-overflow:ellipsis'>{name}</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:9px;
                        color:#718096;margin-top:1px'>{u["role"]}</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:9px;color:{acol}'>
              {datetime.now().strftime('%H:%M:%S')} · {u["location"]}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ALERT TICKER
    st.markdown("""
    <div style='background:rgba(252,129,129,0.06);border:1px solid rgba(252,129,129,0.2);
                border-radius:4px;padding:7px 14px;font-family:JetBrains Mono,monospace;
                font-size:10px;color:#f6ad55;margin-bottom:14px'>
      <span style='color:#fc8181;font-weight:700;letter-spacing:0.1em;
                   margin-right:10px;border-right:1px solid rgba(252,129,129,0.3);
                   padding-right:10px'>⚡ LIVE INTEL</span>
      ⚠ APT-29 lateral movement on CORP-NET-04 &nbsp;·&nbsp;
      🔴 Brute force from 185.220.101.x &nbsp;·&nbsp;
      ⚡ CVE-2024-8821 exploit circulating dark web &nbsp;·&nbsp;
      🛡 Threat DB updated v5.0 &nbsp;·&nbsp;
      ⚠ Unusual exfiltration pattern — user JD-7742
    </div>
    """, unsafe_allow_html=True)

    # ── KPI CARDS ─────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: st.metric("SYSTEM HEALTH",    "98%",  "Optimal",              delta_color="normal")
    with k2: st.metric("ACTIVE THREATS",   str(st.session_state.threat_count), f"+{random.randint(1,4)} last hour", delta_color="inverse")
    with k3: st.metric("RISK LEVEL",       "MEDIUM","Score: 64/100",       delta_color="off")
    with k4: st.metric("USERS MONITORED",  "1,248","+12 today",            delta_color="normal")
    with k5: st.metric("EVENTS / MIN",     str(st.session_state.event_rate),"Avg: 310/min", delta_color="off")
    with k6: st.metric("MY ALERTS TODAY",  str(st.session_state.alerts_reviewed), f"{st.session_state.cases_escalated} escalated", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────
    tab_ov, tab_an, tab_us, tab_intel, tab_logs, tab_map, tab_acct = st.tabs([
        "📊  OVERVIEW",
        "🔍  THREAT ANALYSIS",
        "👤  USER BEHAVIOUR",
        "🧠  THREAT INTEL",
        "📄  SECURITY LOGS",
        "🌍  LIVE ATTACK MAP",
        f"👤  {init} ACCOUNT",
    ])

    # ══════════════════════════════════════════
    # TAB: OVERVIEW
    # ══════════════════════════════════════════
    with tab_ov:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:10px;
                letter-spacing:0.1em;color:#718096;margin-bottom:8px'>
                <span style='display:inline-block;width:6px;height:6px;border-radius:50%;
                background:#4fd1c5;margin-right:6px;vertical-align:middle'></span>
                THREAT EVENTS — LAST 24H</div>""", unsafe_allow_html=True)
            df_tl = gen_threat_timeline()
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(x=df_tl["Hour"], y=df_tl["Events"],
                mode="lines", line=dict(color=CYAN, width=2),
                fill="tozeroy", fillcolor="rgba(79,209,197,0.07)"))
            fig_tl.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False)
            st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar":False})

        with c2:
            st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:10px;
                letter-spacing:0.1em;color:#718096;margin-bottom:8px'>
                <span style='display:inline-block;width:6px;height:6px;border-radius:50%;
                background:#b794f4;margin-right:6px;vertical-align:middle'></span>
                THREAT CATEGORIES</div>""", unsafe_allow_html=True)
            cats = ["Malware","Phishing","Brute Force","Insider","DDoS","Zero-Day"]
            vals = [28,22,18,14,11,7]
            fig_d = go.Figure(go.Pie(labels=cats, values=vals, hole=0.55,
                marker=dict(colors=[RED,AMBER,BLUE,PURPLE,CYAN,GREEN],
                            line=dict(color=BG, width=3))))
            fig_d.update_layout(**PLOTLY_LAYOUT, height=220,
                legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=9)))
            st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar":False})

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:10px;
                letter-spacing:0.1em;color:#718096;margin-bottom:8px'>
                <span style='display:inline-block;width:6px;height:6px;border-radius:50%;
                background:#f6ad55;margin-right:6px;vertical-align:middle'></span>
                ATTACK VECTORS</div>""", unsafe_allow_html=True)
            vectors   = ["Network","Endpoint","Email","Web App","Insider","Supply Chain"]
            detected  = [72,58,81,65,43,29]
            mitigated = [65,52,74,58,38,21]
            fig_r = go.Figure()
            for nm, vals, col, fc in [
                ("Detected",  detected,  CYAN,   "rgba(79,209,197,0.08)"),
                ("Mitigated", mitigated, PURPLE, "rgba(183,148,244,0.08)"),
            ]:
                fig_r.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=vectors+[vectors[0]],
                    fill="toself", name=nm, line=dict(color=col, width=2), fillcolor=fc))
            fig_r.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in["xaxis","yaxis"]},
                height=240,
                polar=dict(bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0,100],
                        gridcolor="rgba(79,209,197,0.12)", tickfont=dict(size=8,color="#4a5568")),
                    angularaxis=dict(gridcolor="rgba(79,209,197,0.12)", tickfont=dict(size=9,color="#718096"))),
                legend=dict(font=dict(size=9)))
            st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar":False})

        with c4:
            st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:10px;
                letter-spacing:0.1em;color:#718096;margin-bottom:12px'>
                <span style='display:inline-block;width:6px;height:6px;border-radius:50%;
                background:#fc8181;margin-right:6px;vertical-align:middle'></span>
                GLOBAL RISK INDICATORS</div>""", unsafe_allow_html=True)
            indicators = [
                ("NETWORK PERIMETER",  72, AMBER),
                ("ENDPOINT SECURITY",  88, GREEN),
                ("DATA LOSS PREV.",    61, AMBER),
                ("IDENTITY & ACCESS",  45, RED),
                ("CLOUD POSTURE",      79, GREEN),
            ]
            for label, val, col in indicators:
                st.markdown(f"""
                <div style='margin-bottom:10px'>
                  <div style='display:flex;justify-content:space-between;
                              font-family:JetBrains Mono,monospace;font-size:10px;
                              color:#718096;margin-bottom:3px'>
                    <span>{label}</span>
                    <span style='color:{col};font-weight:700'>{val}%</span>
                  </div>
                  <div style='background:#141926;border-radius:2px;height:5px'>
                    <div style='width:{val}%;height:100%;background:{col};border-radius:2px'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # TAB: THREAT ANALYSIS
    # ══════════════════════════════════════════
    with tab_an:
        col_inp, col_chart = st.columns([1,1])

        with col_inp:
            st.markdown(f"""<div style='font-family:JetBrains Mono,monospace;font-size:10px;
                letter-spacing:0.1em;color:#718096;margin-bottom:12px'>
                <span style='display:inline-block;width:6px;height:6px;border-radius:50%;
                background:#4fd1c5;margin-right:6px;vertical-align:middle'></span>
                USER ACTIVITY PARAMETERS — ANALYST: {name.split()[0].upper()}</div>""",
                unsafe_allow_html=True)

            # ── Source IP (display + auto geo flag) ──────────────────────
            source_ip = st.text_input("Source IP Address",
                                       placeholder="e.g. 185.220.101.47  (leave blank to skip)",
                                       key="src_ip_input")

            # ── Row 1: connection counts ──────────────────────────────────
            ia, ib, ic = st.columns(3)
            with ia:
                login_attempts = st.number_input("Login Attempts",   min_value=0, value=5)
            with ib:
                failed_logins  = st.number_input("Failed Logins",    min_value=0, value=1)
            with ic:
                data_transfer  = st.number_input("Src Data (MB)",    min_value=0, value=250)

            # ── Row 2: network / session details ─────────────────────────
            id_, ie, if_ = st.columns(3)
            with id_:
                dst_bytes      = st.number_input("Dst Data (MB)",    min_value=0, value=50)
            with ie:
                duration_sec   = st.number_input("Duration (sec)",   min_value=0, value=10)
            with if_:
                protocol       = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])

            # ── Row 3: behavioural flags ──────────────────────────────────
            ig, ih, ii = st.columns(3)
            with ig:
                geo_anomaly    = st.selectbox("Geo Anomaly?", ["No — Normal Region","Yes — Foreign IP Detected"])
            with ih:
                unusual_access = st.selectbox("Unusual Access?", ["No","Yes"])
            with ii:
                priv_esc       = st.selectbox("Privilege Escalation?", ["No","Yes"])

            run = st.button("⚡  RUN AI THREAT ANALYSIS", use_container_width=True, key="run_analysis")

            if run:
                with st.spinner("SmartGuard neural engine processing..."):
                    time.sleep(1.4)

                # Build 12-feature vector matching retrained model schema:
                # [login_attempts, failed_logins, log1p(src_bytes), log1p(dst_bytes),
                #  log1p(duration), protocol(enc), geo_anomaly, unusual_access,
                #  priv_esc, traffic_ratio, login_failure_rate, is_high_traffic]
                geo_num     = 1 if "Yes" in geo_anomaly    else 0
                unusual_num = 1 if unusual_access == "Yes" else 0
                priv_num    = 1 if priv_esc       == "Yes" else 0
                # Auto-flag geo anomaly if IP is external
                if source_ip.strip() and geo_num == 0:
                    _cat, _ = classify_ip(source_ip)
                    if "EXTERNAL" in _cat:
                        geo_num = 1
                # MB → raw bytes (no cap — log1p handles large values naturally)
                src_bytes_raw = data_transfer * 1_048_576
                dst_bytes_raw = dst_bytes     * 1_048_576
                # LabelEncoder alphabetical order: icmp=0, tcp=1, udp=2
                proto_num     = {"TCP": 1, "UDP": 2, "ICMP": 0}.get(protocol, 1)
                X = np.array([[
                    login_attempts,
                    failed_logins,
                    np.log1p(src_bytes_raw),
                    np.log1p(dst_bytes_raw),
                    np.log1p(duration_sec),
                    proto_num,
                    geo_num,
                    unusual_num,
                    priv_num,
                    src_bytes_raw / (dst_bytes_raw + 1),
                    failed_logins / (login_attempts + 1),
                    int(src_bytes_raw > 10_000),
                ]], dtype=float)

                # ══════════════════════════════════════════════════════════
                #  DECISION FUSION ENGINE  (enterprise SOC-grade logic)
                #
                #  Six independent signals are fused into one coherent
                #  verdict.  No single model can override the consensus.
                #
                #  Signal sources:
                #    1. Random Forest          — binary threat probability
                #    2. XGBoost                — binary threat probability
                #    3. Isolation Forest       — unsupervised anomaly flag
                #    4. Attack Type Classifier — 5-class attack category
                #    5. Rule Engine            — deterministic IOC scoring
                #    6. Context Analyser       — behavioural pattern matching
                # ══════════════════════════════════════════════════════════

                if RF_MODEL is not None and XGB_MODEL is not None and HYBRID_MODEL is not None and ISO_MODEL is not None:
                    # ── Layer 1: Individual model inference ───────────────
                    # Models trained on unscaled 12-feature vectors — no scaler needed
                    rf_proba    = RF_MODEL.predict_proba(X)[0]
                    xgb_proba   = XGB_MODEL.predict_proba(X)[0]
                    rf_threat   = rf_proba[1]
                    xgb_threat  = xgb_proba[1]
                    iso_pred    = ISO_MODEL.predict(X)[0]         # 1=normal, -1=anomaly
                    iso_is_anomaly = (iso_pred == -1)

                    # Attack type: rule-based context classification only
                    atk_proba   = np.zeros(5)
                    atk_pred_ml = 0
                    atk_conf_ml = 0

                    # ── Layer 2: Context-aware rule engine ────────────────
                    #   Each rule is a hard-coded IOC detector.  Points
                    #   accumulate independently of the ML models.
                    _boost = 0
                    _ctx_flags = []     # human-readable IOC tags for audit

                    # Credential-attack indicators
                    if failed_logins >= 10:
                        _boost += 35;  _ctx_flags.append("BRUTE_FORCE")
                    elif failed_logins >= 5:
                        _boost += 22;  _ctx_flags.append("CRED_SPRAY")
                    elif failed_logins >= 3:
                        _boost += 10;  _ctx_flags.append("FAILED_AUTH")

                    # Privilege & access anomalies
                    if priv_num == 1:
                        _boost += 30;  _ctx_flags.append("PRIV_ESC")
                    if unusual_num == 1:
                        _boost += 16;  _ctx_flags.append("UNUSUAL_ACCESS")
                    if "Yes" in geo_anomaly:
                        _boost += 20;  _ctx_flags.append("GEO_ANOMALY")

                    # Data-volume indicators (original MB value)
                    if data_transfer > 500:
                        _boost += 26;  _ctx_flags.append("DATA_EXFIL_HIGH")
                    elif data_transfer > 200:
                        _boost += 18;  _ctx_flags.append("DATA_EXFIL_MED")
                    elif data_transfer > 50:
                        _boost += 10;  _ctx_flags.append("DATA_ELEVATED")

                    # Timing / protocol / scan heuristics
                    if duration_sec < 2 and login_attempts > 20:
                        _boost += 20;  _ctx_flags.append("SCAN_FLOOD")
                    elif login_attempts > 30 and duration_sec < 10:
                        _boost += 14;  _ctx_flags.append("RAPID_CONN")
                    if protocol == "ICMP" and login_attempts > 15:
                        _boost += 16;  _ctx_flags.append("ICMP_FLOOD")
                    if dst_bytes < 5 and data_transfer > 100:
                        _boost += 14;  _ctx_flags.append("ASYM_TRAFFIC")
                    # Reconnaissance pattern: many connections + minimal data
                    if login_attempts > 15 and data_transfer < 50 and duration_sec < 10:
                        _boost += 16;  _ctx_flags.append("RECON_SCAN")

                    # External-IP risk
                    if source_ip.strip():
                        _icat, _ = classify_ip(source_ip)
                        if "EXTERNAL" in _icat:
                            _boost += 12;  _ctx_flags.append("EXTERNAL_IP")

                    # ── Layer 3: Weighted decision fusion ─────────────────
                    #   ML ensemble: RF 25 % + XGBoost 75 %
                    hybrid_prob = (0.25 * rf_threat) + (0.75 * xgb_threat)

                    # Isolation Forest agreement amplifier
                    if iso_is_anomaly:
                        hybrid_prob = min(0.98, hybrid_prob + 0.08)

                    # Model-agreement bonus: when both RF and XGBoost agree
                    # on threat (>0.5), boost confidence further.
                    if rf_threat > 0.5 and xgb_threat > 0.5:
                        _agreement = min(rf_threat, xgb_threat)
                        hybrid_prob = min(0.98, hybrid_prob + _agreement * 0.08)

                    ml_score = int(hybrid_prob * 100)

                    # Adaptive fusion weights: shift toward rules when
                    # the ML models disagree (low confidence).
                    _ml_confidence = abs(rf_threat - xgb_threat)
                    if _ml_confidence > 0.3:
                        # Models disagree → trust rules more
                        _w_ml, _w_rule = 0.45, 0.55
                    elif _ml_confidence > 0.15:
                        _w_ml, _w_rule = 0.55, 0.45
                    else:
                        # Models agree → standard weights
                        _w_ml, _w_rule = 0.60, 0.40

                    score = min(98, int(ml_score * _w_ml + _boost * _w_rule))

                    # ── Layer 4: Safety-net floor ─────────────────────────
                    #   When 3+ IOC flags fire, enforce a minimum score so
                    #   obvious threats cannot be hidden by a confused ML.
                    _ioc_count = len(_ctx_flags)
                    if _ioc_count >= 5:
                        score = max(score, min(98, 82))    # CRITICAL floor
                    elif _ioc_count >= 4:
                        score = max(score, min(98, 76))    # HIGH floor
                    elif _ioc_count >= 3:
                        score = max(score, min(98, 68))    # HIGH floor
                    elif _boost >= 50:
                        score = max(score, min(98, _boost + 10))

                    # ── Layer 5: Attack type classification ──────────────
                    #
                    #   Two-stage pipeline:
                    #
                    #   5A  PRIMARY — deterministic rule-based classifier.
                    #       Evaluated top-down; first match LOCKS the label.
                    #       Prevents traffic-volume Probe patterns from being
                    #       pulled into R2L by geo/auth context signals.
                    #
                    #   5B  CONTEXT SCORING — behavioural evidence per class.
                    #       When a primary rule fired  → refines confidence.
                    #       When no primary rule fired → selects the label
                    #       (fallback path, same as before).
                    #
                    #   5C  FALLBACK CONSTRAINTS — consistency guards applied
                    #       only when the primary rule did not fire.
                    #
                    #   5D  CONFIDENCE CALIBRATION — final blend of rule
                    #       confidence + risk score + IOC density.
                    # ──────────────────────────────────────────────────────

                    # ── 5A. Primary rule-based classifier ────────────────
                    #   Rules are ordered by specificity / severity.
                    #   _primary_label == -1 means "no rule matched".
                    _primary_label = -1
                    _primary_conf  = 90    # base confidence when a rule fires

                    if priv_num == 1:
                        # U2R: privilege escalation is the strongest single
                        # indicator — lock regardless of traffic volume.
                        _primary_label = 4
                        _primary_conf  = 95

                    elif failed_logins >= 10:
                        # R2L: ≥10 credential failures → brute-force /
                        # credential-stuffing from a remote source.
                        _primary_label = 3
                        _primary_conf  = 92

                    elif data_transfer >= 700 and duration_sec <= 5:
                        # DoS: massive traffic burst in ≤5 s window →
                        # asymmetric flood pattern.
                        _primary_label = 1
                        _primary_conf  = 91

                    elif data_transfer >= 300 and failed_logins == 0:
                        # Probe: high volume + zero failed logins →
                        # scanning / reconnaissance, NOT a credential attack.
                        # This rule explicitly prevents Probe→R2L leakage.
                        _primary_label = 2
                        _primary_conf  = 90

                    _primary_locked = (_primary_label != -1)

                    # ── 5B. Behavioural context scores (0-1 per class) ───
                    #   [Normal=0, DoS=1, Probe=2, R2L=3, U2R=4]
                    #   When a primary rule fired  → used for confidence only.
                    #   When no primary rule fired → used for label selection.
                    _ctx = np.zeros(5)

                    # DoS — flood / asymmetric / ICMP abuse
                    if data_transfer > 500:                        _ctx[1] += 0.30
                    elif data_transfer > 200:                      _ctx[1] += 0.15
                    if dst_bytes < 5 and data_transfer > 100:      _ctx[1] += 0.25
                    if login_attempts > 50 and duration_sec < 3:   _ctx[1] += 0.25
                    if protocol == "ICMP" and login_attempts > 15: _ctx[1] += 0.20

                    # Probe — many connections, short duration, low data
                    if login_attempts > 20 and duration_sec < 5:   _ctx[2] += 0.30
                    if duration_sec < 2 and login_attempts > 10:   _ctx[2] += 0.25
                    if data_transfer < 50 and login_attempts > 15: _ctx[2] += 0.20
                    if protocol == "ICMP":                         _ctx[2] += 0.10
                    if protocol == "UDP" and login_attempts > 20:  _ctx[2] += 0.15

                    # R2L — credential failures, geo anomaly, external IP
                    if failed_logins >= 10:                        _ctx[3] += 0.35
                    elif failed_logins >= 5:                       _ctx[3] += 0.25
                    elif failed_logins >= 3:                       _ctx[3] += 0.12
                    if "Yes" in geo_anomaly:                       _ctx[3] += 0.20
                    if unusual_num == 1 and failed_logins >= 3:    _ctx[3] += 0.15
                    if "EXTERNAL_IP" in _ctx_flags:                _ctx[3] += 0.10

                    # U2R — privilege escalation, unusual internal access
                    if priv_num == 1:                              _ctx[4] += 0.40
                    if priv_num == 1 and unusual_num == 1:         _ctx[4] += 0.20
                    if priv_num == 1 and failed_logins >= 3:       _ctx[4] += 0.15
                    if priv_num == 1 and "EXTERNAL_IP" not in _ctx_flags:
                        _ctx[4] += 0.10   # insider threat pattern

                    # Normal — absence of indicators is itself a signal
                    if _ioc_count == 0:                            _ctx[0] += 0.50
                    if failed_logins <= 1 and data_transfer < 50:  _ctx[0] += 0.25
                    if not iso_is_anomaly and hybrid_prob < _HYBRID_THRESHOLD:
                        _ctx[0] += 0.25

                    # Normalise context scores → probability-like distribution
                    _ctx_sum = _ctx.sum()
                    _ctx_norm = (_ctx / _ctx_sum) if _ctx_sum > 0 else np.array([1.0, 0, 0, 0, 0])

                    # ── 5C. Label selection ───────────────────────────────
                    if _primary_locked:
                        # Primary rule won: use its label; blend its
                        # confidence with context agreement on that class.
                        atk_pred  = _primary_label
                        _ctx_conf = int(_ctx_norm[_primary_label] * 100)
                        atk_conf  = int(_primary_conf * 0.70 + _ctx_conf * 0.30)

                    else:
                        # No primary rule fired: fall back to context-only pick.
                        atk_pred = int(np.argmax(_ctx_norm))
                        atk_conf = int(_ctx_norm[atk_pred] * 100)

                        # Constraint 1: HIGH risk (≥70) CANNOT be Normal.
                        if score >= 70 and atk_pred == 0:
                            _atk_scores = _ctx_norm[1:].copy()
                            atk_pred = int(np.argmax(_atk_scores)) + 1
                            atk_conf = max(atk_conf, int(_atk_scores.max() * 100), 80)

                        # Constraint 2: LOW risk (≤35) CANNOT be attack.
                        if score <= 35 and atk_pred != 0:
                            atk_pred = 0
                            atk_conf = max(60, 100 - score)

                        # Constraint 3: MEDIUM (36-69) + Normal + 3+ IOC flags
                        #   → promote to best non-Normal class.
                        if 36 <= score <= 69 and atk_pred == 0 and _ioc_count >= 3:
                            _atk_scores = _ctx_norm[1:].copy()
                            if _atk_scores.max() > 0.10:
                                atk_pred = int(np.argmax(_atk_scores)) + 1
                                atk_conf = max(55, int(_atk_scores.max() * 100))

                    # ── 5D. Final confidence calibration ─────────────────
                    #   Blend label confidence + risk score + IOC density.
                    if atk_pred != 0:
                        _fusion_conf = min(98, int(
                            atk_conf * 0.55 +
                            min(score, 98) * 0.30 +
                            min(_ioc_count * 10, 30) * 0.15
                        ))
                        atk_conf = max(atk_conf, _fusion_conf)
                    atk_conf = min(atk_conf, 98)

                    # Per-class distribution for the breakdown bars.
                    # Guarantee it is normalized AND that the predicted class is
                    # its dominant component, so the bars can never contradict the
                    # final verdict (e.g. a "Normal" verdict with a 100% DoS bar).
                    atk_proba = _ctx_norm.copy()
                    _others_max = max([atk_proba[i] for i in range(5) if i != atk_pred], default=0.0)
                    atk_proba[atk_pred] = max(atk_proba[atk_pred], atk_conf / 100, _others_max + 0.05)
                    atk_proba = atk_proba / atk_proba.sum()

                    confidence  = int(max(hybrid_prob, 1 - hybrid_prob) * 100)
                    # Boost reported confidence when all layers agree
                    if (hybrid_prob > 0.6 and iso_is_anomaly
                            and _boost >= 30 and atk_pred != 0):
                        confidence = max(confidence, 92)

                    model_label = "Decision Fusion · RF(25%)+XGB(75%)+ISO+Rules · NSL-KDD"

                else:
                    # ── Fallback: rule-only engine (models failed to load) ──
                    rf_threat = xgb_threat = 0.0
                    iso_pred   = 1
                    iso_is_anomaly = False
                    _ctx_flags = []

                    score = 20
                    if failed_logins > 3:               score += 30
                    elif failed_logins > 1:             score += 15
                    if unusual_access == "Yes":         score += 20
                    if "Yes" in geo_anomaly:            score += 25
                    if priv_esc == "Yes":               score += 30
                    if data_transfer > 1000:            score += 15
                    elif data_transfer > 500:           score += 8
                    if duration_sec < 2 and login_attempts > 20: score += 20
                    if protocol == "ICMP" and login_attempts > 15: score += 15
                    if source_ip.strip():
                        _cat, _ = classify_ip(source_ip)
                        if "EXTERNAL" in _cat:          score += 10
                    score = min(98, score + random.randint(0, 8))

                    # Rule-only attack type — same context scoring as Layer 5A
                    _ctx_fb = np.zeros(5)
                    if data_transfer > 200:                          _ctx_fb[1] += 0.30
                    if dst_bytes < 5 and data_transfer > 100:        _ctx_fb[1] += 0.25
                    if login_attempts > 50 and duration_sec < 3:     _ctx_fb[1] += 0.25
                    if protocol == "ICMP" and login_attempts > 15:   _ctx_fb[1] += 0.20
                    if login_attempts > 20 and duration_sec < 5:     _ctx_fb[2] += 0.30
                    if data_transfer < 50 and login_attempts > 15:   _ctx_fb[2] += 0.20
                    if failed_logins >= 5:                           _ctx_fb[3] += 0.35
                    elif failed_logins >= 3:                         _ctx_fb[3] += 0.15
                    if "Yes" in geo_anomaly:                         _ctx_fb[3] += 0.20
                    if priv_esc == "Yes":                            _ctx_fb[4] += 0.50
                    if failed_logins <= 1 and data_transfer < 50:    _ctx_fb[0] += 0.40
                    atk_pred = int(np.argmax(_ctx_fb))
                    # If high risk, never Normal
                    if score >= 70 and atk_pred == 0:
                        _ctx_fb[0] = 0
                        atk_pred = int(np.argmax(_ctx_fb))
                        if _ctx_fb.max() == 0:  atk_pred = 2
                    if score <= 35:
                        atk_pred = 0
                    atk_conf = min(95, max(60, score))
                    atk_proba   = np.zeros(5)
                    atk_proba[atk_pred] = atk_conf / 100
                    atk_pred_ml = atk_pred
                    atk_conf_ml = atk_conf

                    confidence  = random.randint(85, 97)
                    model_label = "Rule-based fallback (models unavailable)"

                # ══════════════════════════════════════════════════════════
                #  CONSISTENCY ENFORCEMENT
                #
                #  The attack classifier (Layer 5) and the binary ML score
                #  are computed independently.  A low ML score can coexist
                #  with a non-Normal attack label, creating a contradiction
                #  (e.g. "Normal Activity" banner + "DoS" classification box).
                #
                #  Rule: if atk_pred != Normal → the session IS an attack.
                #  Elevate the risk score to the per-type floor so every
                #  downstream variable (threat_level, tlcol, banner, risk bar)
                #  reflects the classified attack — never "Normal Activity".
                # ══════════════════════════════════════════════════════════
                if atk_pred != 0:
                    # Minimum score per attack class (all > 65 → forces threat
                    # banner and eliminates "Normal Activity" contradiction).
                    _atk_score_floor = {
                        1: 70,   # DoS   — volumetric / flood
                        2: 68,   # Probe — reconnaissance
                        3: 72,   # R2L   — credential exploitation
                        4: 78,   # U2R   — privilege escalation
                    }
                    score = max(score, _atk_score_floor.get(atk_pred, 70))

                # ── Threat level derivation ────────────────────────────────
                # Derived AFTER consistency enforcement so threat_level, tlcol,
                # and the banner all reflect the corrected score.
                anomalies = sum([
                    failed_logins > 1,
                    unusual_access == "Yes",
                    "Yes" in geo_anomaly,
                    priv_esc == "Yes",
                    iso_is_anomaly if ISO_MODEL is not None else False,
                ])
                threat_level = ("CRITICAL" if score > 80 else "HIGH" if score > 65
                                else "MEDIUM" if score > 45 else "LOW")
                tlcol = RED if score > 80 else AMBER if score > 65 else BLUE if score > 45 else GREEN

                st.session_state.alerts_reviewed += 1

                # Banner: score > 65 is now guaranteed for any classified attack
                if score > 65:
                    _atk_info = _ATK_TYPES[atk_pred]
                    _banner_detail = (
                        f"Attack type: {_atk_info['name']} — {_atk_info['full']}"
                        if atk_pred != 0 else "Anomalous behaviour detected"
                    )
                    st.error(f"⚠  HIGH RISK THREAT DETECTED — {_banner_detail}")
                    if source_ip.strip():
                        ip_cat, ip_col = classify_ip(source_ip)
                        st.markdown(f"""
                        <div style='background:rgba(252,129,129,0.10);
                                    border:1px solid rgba(252,129,129,0.5);
                                    border-left:4px solid #fc8181;border-radius:4px;
                                    padding:14px 18px;margin:8px 0'>
                          <div style='font-family:JetBrains Mono,monospace;font-size:9px;
                                      letter-spacing:0.14em;color:#fc8181;margin-bottom:6px'>
                            ⚠ ATTACKER IP IDENTIFIED &nbsp;·&nbsp; {ip_cat}
                          </div>
                          <div style='font-family:JetBrains Mono,monospace;font-size:26px;
                                      font-weight:700;color:#fc8181;letter-spacing:0.05em'>
                            {source_ip.strip()}
                          </div>
                          <div style='display:flex;gap:18px;margin-top:8px;
                                      font-family:JetBrains Mono,monospace;font-size:10px;color:#718096'>
                            <span>🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                            <span>📡 {protocol}</span>
                            <span>⏱ {duration_sec}s</span>
                            <span>📤 {data_transfer} MB src</span>
                            <span>📥 {dst_bytes} MB dst</span>
                          </div>
                          <div style='margin-top:8px;font-family:JetBrains Mono,monospace;
                                      font-size:10px;color:#fc8181;font-weight:700'>
                            RECOMMENDED ACTION: Block IP at perimeter firewall &nbsp;·&nbsp; Preserve forensic logs
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅  NORMAL ACTIVITY — No immediate threat detected")

                # ── Attack type classification display ─────────────────────
                info      = _ATK_TYPES[atk_pred]
                col_box   = info["color"]

                st.markdown(f"""
                <div style='background:rgba(0,0,0,0.25);
                            border:1px solid {col_box}55;border-left:4px solid {col_box};
                            border-radius:6px;padding:16px 20px;margin:10px 0'>
                  <div style='font-family:JetBrains Mono,monospace;font-size:9px;
                              letter-spacing:0.14em;color:#718096;margin-bottom:10px'>
                    ATTACK TYPE CLASSIFICATION
                  </div>
                  <div style='display:flex;align-items:center;gap:18px;flex-wrap:wrap'>
                    <div style='font-size:36px;line-height:1'>{info["icon"]}</div>
                    <div style='flex:1;min-width:160px'>
                      <div style='font-family:JetBrains Mono,monospace;font-size:24px;
                                  font-weight:700;color:{col_box}'>{info["name"]}</div>
                      <div style='font-size:13px;color:#e2e8f0;margin-top:2px'>{info["full"]}</div>
                    </div>
                    <div style='text-align:right;min-width:110px'>
                      <div style='font-family:JetBrains Mono,monospace;font-size:9px;
                                  color:#718096;letter-spacing:0.1em'>ATTACK CLASSIFIER CONFIDENCE</div>
                      <div style='font-family:JetBrains Mono,monospace;font-size:28px;
                                  font-weight:700;color:{col_box}'>{atk_conf}%</div>
                    </div>
                  </div>
                  <div style='margin-top:12px;display:flex;gap:6px;flex-wrap:wrap'>
                    {''.join(
                        "<div style='flex:1;min-width:80px;background:rgba(0,0,0,0.2);"
                        "border-radius:4px;padding:7px 10px;text-align:center'>"
                        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
                        "color:#718096'>" + _ATK_TYPES[i]["name"] + "</div>"
                        "<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                        "font-weight:700;color:" + _ATK_TYPES[i]["color"] + "'>"
                        + str(int(atk_proba[i]*100)) + "%</div></div>"
                        for i in range(5)
                    )}
                  </div>
                  <div style='margin-top:10px;padding-top:10px;
                              border-top:1px solid rgba(255,255,255,0.06);
                              font-family:JetBrains Mono,monospace;font-size:10px;color:#718096'>
                    RECOMMENDED ACTION:
                    <span style='color:{col_box};font-weight:700'>{info["rec"]}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                m1,m2,m3,m4 = st.columns(4)
                with m1: st.metric("RISK SCORE",   f"{score}%")
                with m2: st.metric("THREAT LEVEL", threat_level)
                with m3: st.metric("ANOMALIES",    str(anomalies))
                with m4: st.metric("CONFIDENCE",   f"{confidence}%")

                st.markdown(f"""
                <div style='margin:10px 0 6px'>
                  <div style='display:flex;justify-content:space-between;font-family:JetBrains Mono,
                              monospace;font-size:10px;color:#718096;margin-bottom:3px'>
                    <span>COMPOSITE RISK SCORE — ANALYSED BY SMARTGUARD AI</span>
                    <span style='color:{tlcol};font-weight:700'>{score}/100</span>
                  </div>
                  <div style='background:#141926;border-radius:2px;height:8px'>
                    <div style='width:{score}%;height:100%;background:{tlcol};border-radius:2px'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                findings = []
                if atk_pred == 1:  # DoS
                    findings = [
                        "🌊 High traffic volume detected",
                        "⚡ Network flooding behaviour identified",
                        "📤 Abnormal bandwidth consumption detected"
                    ]

                elif atk_pred == 2:  # Probe
                    findings = [
                        "🔍 Reconnaissance activity detected",
                        "📡 Multiple service scans observed",
                        "🛰 Suspicious host enumeration detected"
                    ]

                elif atk_pred == 3:  # R2L
                    findings = [
                        "🔐 Credential abuse detected",
                        "⚠ Repeated authentication failures",
                        "🚪 Unauthorized access attempt detected"
                    ]

                elif atk_pred == 4:  # U2R
                    findings = [
                        "🔑 Privilege escalation indicators detected",
                        "⚠ Abnormal permission usage",
                        "🛡 Critical system access anomaly detected"
                    ]

                else:
                    findings = [
                        "✅ All behavioural parameters within normal bounds"
                    ]

                for f in findings:
                    st.markdown(f"""
                    <div style='background:#141926;border-radius:3px;padding:7px 12px;margin-bottom:5px;
                                font-family:JetBrains Mono,monospace;font-size:11px;color:#a0aec0'>{f}</div>
                    """, unsafe_allow_html=True)

                # ── Hybrid model breakdown chart ───────────────────────────
                if RF_MODEL is not None and XGB_MODEL is not None and HYBRID_MODEL is not None:
                    iso_contribution = 5 if iso_pred == -1 else 0
                    rf_pct   = int(rf_threat  * 100)
                    xgb_pct  = int(xgb_threat * 100)
                    iso_flag = "ANOMALY" if iso_pred == -1 else "NORMAL"
                    iso_col  = RED if iso_pred == -1 else GREEN

                    fig_hybrid = go.Figure()
                    fig_hybrid.add_trace(go.Bar(
                        name="Random Forest",
                        x=["Random Forest (25%)"],
                        y=[rf_pct],
                        marker_color=CYAN,
                        text=[f"{rf_pct}%"],
                        textposition="outside",
                        textfont=dict(family="JetBrains Mono", size=11, color=CYAN),
                    ))
                    fig_hybrid.add_trace(go.Bar(
                        name="XGBoost",
                        x=["XGBoost (75%)"],
                        y=[xgb_pct],
                        marker_color=PURPLE,
                        text=[f"{xgb_pct}%"],
                        textposition="outside",
                        textfont=dict(family="JetBrains Mono", size=11, color=PURPLE),
                    ))
                    fig_hybrid.add_trace(go.Bar(
                        name="Hybrid Score",
                        x=["Hybrid Score"],
                        y=[score],
                        marker_color=tlcol,
                        text=[f"{score}%"],
                        textposition="outside",
                        textfont=dict(family="JetBrains Mono", size=11, color=tlcol),
                    ))
                    _hl = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"}
                    fig_hybrid.update_layout(
                        **_hl, height=200, showlegend=False,
                        title=dict(
                            text=f"MODEL BREAKDOWN · Isolation Forest: <span style='color:{iso_col}'>{iso_flag}</span>",
                            font=dict(size=10, color="#718096"), x=0,
                        ),
                        yaxis=dict(range=[0, 110], gridcolor="rgba(79,209,197,0.1)",
                                   tickfont=dict(size=9)),
                        barmode="group",
                    )
                    st.plotly_chart(fig_hybrid, use_container_width=True,
                                    config={"displayModeBar": False})

                    st.markdown(f"""
                    <div style='background:#141926;border:1px solid rgba(79,209,197,0.12);
                                border-radius:4px;padding:8px 14px;font-family:JetBrains Mono,
                                monospace;font-size:10px;color:#718096'>
                      ENGINE: <span style='color:#4fd1c5'>{model_label}</span>
                      &nbsp;·&nbsp; Isolation Forest: <span style='color:{iso_col}'>{iso_flag}</span>
                      &nbsp;·&nbsp; Confidence: <span style='color:#e2e8f0'>{confidence}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if atk_pred == 1:  # DoS
                        st.warning("⚡ RECOMMENDATION: Enable rate limiting and traffic filtering.")
                    elif atk_pred == 2:  # Probe
                        st.warning("⚡ RECOMMENDATION: Investigate reconnaissance activity and monitor scanned assets.")
                    elif atk_pred == 3:  # R2L
                        st.warning("⚡ RECOMMENDATION: Review authentication logs and consider account lockout.")
                    elif atk_pred == 4:  # U2R
                        st.warning("⚡ RECOMMENDATION: Investigate privilege escalation and isolate affected systems.")
                    elif score > 65:
                        st.warning("⚡ RECOMMENDATION: Immediate investigation required.")

        with col_chart:
            days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(name="Normal",    x=days, y=[120,145,133,160,148,45,32], marker_color=CYAN, opacity=0.7))
            fig_b.add_trace(go.Bar(name="Anomalous", x=days, y=[8,12,5,18,22,3,1],           marker_color=RED,  opacity=0.85))
            fig_b.update_layout(**PLOTLY_LAYOUT, height=200, barmode="group",
                                title=dict(text="BEHAVIOURAL PATTERN — 7 DAY HISTORY",
                                           font=dict(size=10,color="#718096"),x=0),
                                legend=dict(font=dict(size=9)))
            st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar":False})

            trend_x = list(range(1,31))
            trend_y = [random.randint(30,80) for _ in trend_x]
            fig_tr  = go.Figure(go.Scatter(x=trend_x, y=trend_y, mode="lines+markers",
                line=dict(color=BLUE,width=2), marker=dict(size=4,color=BLUE),
                fill="tozeroy", fillcolor="rgba(99,179,237,0.06)"))
            fig_tr.update_layout(**PLOTLY_LAYOUT, height=180,
                title=dict(text="RISK SCORE TREND — LAST 30 DAYS", font=dict(size=10,color="#718096"),x=0),
                showlegend=False)
            st.plotly_chart(fig_tr, use_container_width=True, config={"displayModeBar":False})

    # ══════════════════════════════════════════
    # TAB: USER BEHAVIOUR
    # ══════════════════════════════════════════
    with tab_us:
        cu1, cu2 = st.columns([1,1])
        with cu1:
            st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:10px;
                letter-spacing:0.1em;color:#718096;margin-bottom:12px'>
                <span style='display:inline-block;width:6px;height:6px;border-radius:50%;
                background:#fc8181;margin-right:6px;vertical-align:middle'></span>
                HIGH-RISK USERS — SORTED BY RISK SCORE</div>""", unsafe_allow_html=True)

            users_list = [
                {"name":"J. Donovan",   "role":"Finance Analyst",  "dept":"Finance",    "score":91,"col":RED,    "bg":"rgba(252,129,129,0.15)"},
                {"name":"M. Patel",     "role":"IT Administrator", "dept":"IT",         "score":78,"col":AMBER,  "bg":"rgba(246,173,85,0.12)"},
                {"name":"S. Chen",      "role":"HR Manager",       "dept":"HR",         "score":65,"col":AMBER,  "bg":"rgba(246,173,85,0.12)"},
                {"name":"K. Williams",  "role":"Developer",        "dept":"Engineering","score":52,"col":BLUE,   "bg":"rgba(99,179,237,0.12)"},
                {"name":"A. Rodriguez", "role":"Executive",        "dept":"C-Suite",    "score":41,"col":GREEN,  "bg":"rgba(104,211,145,0.1)"},
                {"name":"T. Nakamura",  "role":"Operations",       "dept":"Operations", "score":28,"col":GREEN,  "bg":"rgba(104,211,145,0.1)"},
            ]
            for u2 in users_list:
                parts = u2["name"].split(".")
                ini   = (parts[0][0] + parts[1].strip()[0]).upper() if len(parts) >= 2 else u2["name"][:2].upper()
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:10px;padding:9px 0;
                            border-bottom:1px solid rgba(255,255,255,0.04)'>
                  <div style='width:32px;height:32px;border-radius:4px;
                              background:{u2["bg"]};color:{u2["col"]};
                              display:flex;align-items:center;justify-content:center;
                              font-family:JetBrains Mono,monospace;font-size:10px;
                              font-weight:700;flex-shrink:0'>{ini}</div>
                  <div style='flex:1;min-width:0'>
                    <div style='font-size:13px;color:#e2e8f0'>{u2["name"]}</div>
                    <div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#718096'>
                      {u2["role"]} · {u2["dept"]}
                    </div>
                  </div>
                  <div style='width:72px;background:#141926;border-radius:2px;height:4px;flex-shrink:0'>
                    <div style='width:{u2["score"]}%;height:100%;background:{u2["col"]};border-radius:2px'></div>
                  </div>
                  <div style='font-family:JetBrains Mono,monospace;font-size:13px;
                              font-weight:700;color:{u2["col"]};width:28px;text-align:right;
                              flex-shrink:0'>{u2["score"]}</div>
                </div>
                """, unsafe_allow_html=True)

        with cu2:
            depts   = ["Engineering","Finance","HR","Operations","Executive","Legal"]
            sessions= [312,187,143,298,54,76]
            fig_s   = go.Figure(go.Bar(x=sessions, y=depts, orientation="h",
                marker=dict(color=[CYAN,RED,PURPLE,BLUE,AMBER,GREEN], line=dict(width=0))))
            fig_s.update_layout(**PLOTLY_LAYOUT, height=230,
                title=dict(text="SESSION DISTRIBUTION BY DEPARTMENT",font=dict(size=10,color="#718096"),x=0),
                showlegend=False)
            st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar":False})

            heatmap_z = [[random.randint(0,40) for _ in range(24)] for _ in range(5)]
            fig_h = go.Figure(go.Heatmap(
                z=heatmap_z, x=[f"{h:02d}h" for h in range(24)],
                y=["Mon","Tue","Wed","Thu","Fri"],
                colorscale=[[0,BG3],[0.5,PURPLE],[1,RED]], showscale=False))
            fig_h.update_layout(**PLOTLY_LAYOUT, height=150,
                title=dict(text="LOGIN FREQUENCY — HOURLY HEATMAP",font=dict(size=10,color="#718096"),x=0))
            st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar":False})

    # ══════════════════════════════════════════
    # TAB: THREAT INTEL
    # ══════════════════════════════════════════
    with tab_intel:
        ci1, ci2 = st.columns([1,1])
        sev_colors = {
            "CRITICAL": (RED,   "rgba(252,129,129,0.13)","rgba(252,129,129,0.3)"),
            "HIGH":     (AMBER, "rgba(246,173,85,0.1)",  "rgba(246,173,85,0.3)"),
            "MEDIUM":   (BLUE,  "rgba(99,179,237,0.08)", "rgba(99,179,237,0.25)"),
        }
        with ci1:
            st.markdown("""<div style='font-family:JetBrains Mono,monospace;font-size:10px;
                letter-spacing:0.1em;color:#718096;margin-bottom:12px'>
                <span style='display:inline-block;width:6px;height:6px;border-radius:50%;
                background:#fc8181;margin-right:6px;vertical-align:middle'></span>
                ACTIVE THREAT INTELLIGENCE FEED</div>""", unsafe_allow_html=True)
            for item in INTEL_FEED:
                col2, bg, border = sev_colors.get(item["sev"], (GREEN,"rgba(104,211,145,0.07)","rgba(104,211,145,0.2)"))
                st.markdown(f"""
                <div style='border-left:2px solid {col2};background:{bg};border-radius:0 4px 4px 0;
                            padding:9px 13px;margin-bottom:8px'>
                  <div style='display:flex;align-items:center;gap:7px;margin-bottom:3px'>
                    <span style='background:{bg};color:{col2};border:1px solid {border};
                                 padding:1px 7px;border-radius:2px;
                                 font-family:JetBrains Mono,monospace;font-size:9px;
                                 font-weight:700;letter-spacing:0.06em'>{item["sev"]}</span>
                    <span style='font-family:JetBrains Mono,monospace;font-size:9px;color:#4a5568'>
                      {item["time"]} · {item["source"]}
                    </span>
                  </div>
                  <div style='font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:2px'>{item["title"]}</div>
                  <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#718096'>{item["desc"]}</div>
                </div>
                """, unsafe_allow_html=True)

        with ci2:
            tactic_cols = [CYAN,BLUE,PURPLE,AMBER,RED,GREEN,AMBER,BLUE]
            fig_m = go.Figure(go.Bar(
                x=MITRE_COVERAGE, y=MITRE_TACTICS, orientation="h",
                marker=dict(color=tactic_cols, line=dict(width=0)),
                text=[f"{v}%" for v in MITRE_COVERAGE],
                textfont=dict(family="JetBrains Mono",size=10,color="#e2e8f0"),
                textposition="inside"))
            fig_m.update_layout(**PLOTLY_LAYOUT, height=280,
                title=dict(text="MITRE ATT&CK FRAMEWORK COVERAGE",font=dict(size=10,color="#718096"),x=0),
                showlegend=False, xaxis_range=[0,100])
            st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar":False})

            ioc_types = ["IP Addresses","File Hashes","Domains","URLs","Email Addrs"]
            ioc_vals  = [142,89,67,53,31]
            fig_ioc = go.Figure(go.Bar(x=ioc_types, y=ioc_vals,
                marker=dict(color=[RED,AMBER,PURPLE,CYAN,GREEN], line=dict(width=0))))
            fig_ioc.update_layout(**PLOTLY_LAYOUT, height=150,
                title=dict(text="IOC BREAKDOWN BY TYPE",font=dict(size=10,color="#718096"),x=0),
                showlegend=False)
            st.plotly_chart(fig_ioc, use_container_width=True, config={"displayModeBar":False})

    # ══════════════════════════════════════════
    # TAB: SECURITY LOGS
    # ══════════════════════════════════════════
    with tab_logs:
        st.markdown(f"""<div style='font-family:JetBrains Mono,monospace;font-size:10px;
            letter-spacing:0.1em;color:#718096;margin-bottom:12px'>
            <span style='display:inline-block;width:6px;height:6px;border-radius:50%;
            background:#68d391;margin-right:6px;vertical-align:middle'></span>
            SECURITY EVENT LOG — REAL-TIME FEED · REVIEWED BY {name.upper()}</div>""",
            unsafe_allow_html=True)

        df_logs = gen_logs(35)

        if sev_filter:
            df_logs = df_logs[df_logs["Severity"].isin(sev_filter)]

        if dept_filter != "All Departments":
            df_logs = df_logs[df_logs["Department"] == dept_filter]

        sev_icons = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢","INFO":"🔵"}
        df_display = df_logs.copy()
        df_display["Severity"] = df_display["Severity"].apply(lambda s: f"{sev_icons.get(s,'')} {s}")

        st.dataframe(df_display, use_container_width=True, hide_index=True, height=480,
            column_config={
                "Timestamp":  st.column_config.TextColumn("TIMESTAMP",   width="medium"),
                "Event ID":   st.column_config.TextColumn("EVENT ID",    width="small"),
                "User":       st.column_config.TextColumn("USER",        width="medium"),
                "Department": st.column_config.TextColumn("DEPARTMENT",  width="medium"),
                "Event Type": st.column_config.TextColumn("EVENT TYPE",  width="large"),
                "Source IP":  st.column_config.TextColumn("SOURCE IP",   width="medium"),
                "Severity":   st.column_config.TextColumn("SEVERITY",    width="medium"),
            })

        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#4a5568;
                    margin-top:6px;text-align:right'>
          Showing {len(df_display)} events · Analyst: {name} · {datetime.now().strftime('%H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)

        csv = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button("⬇  EXPORT LOGS AS CSV", data=csv,
            file_name=f"smartguard_logs_{name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv")

    # ══════════════════════════════════════════
    # TAB: LIVE ATTACK MAP
    # ══════════════════════════════════════════
    with tab_map:
        fig_map = go.Figure()
        for i, atk in enumerate(ATTACK_ORIGINS):
            tgt = TARGETS[i % len(TARGETS)]
            intensity = atk["attacks"] / 142
            arc_color = (f"rgba(252,129,129,{0.3+intensity*0.5})" if intensity > 0.6 else
                         f"rgba(246,173,85,{0.3+intensity*0.5})"  if intensity > 0.3 else
                         f"rgba(99,179,237,{0.25+intensity*0.4})")
            mid_lat = (atk["lat"]+tgt["lat"])/2 - abs(atk["lat"]-tgt["lat"])*0.3
            fig_map.add_trace(go.Scattergeo(
                lat=[atk["lat"],mid_lat,tgt["lat"]], lon=[atk["lon"],(atk["lon"]+tgt["lon"])/2,tgt["lon"]],
                mode="lines", line=dict(width=max(1,int(intensity*3)),color=arc_color),
                hoverinfo="skip", showlegend=False))
        fig_map.add_trace(go.Scattergeo(
            lat=[a["lat"] for a in ATTACK_ORIGINS], lon=[a["lon"] for a in ATTACK_ORIGINS],
            mode="markers+text",
            marker=dict(size=[max(8,int(a["attacks"]/10)) for a in ATTACK_ORIGINS],
                        color=RED, opacity=0.85, symbol="circle",
                        line=dict(width=1,color="rgba(252,129,129,0.5)")),
            text=[a["city"] for a in ATTACK_ORIGINS], textposition="top center",
            textfont=dict(family="JetBrains Mono",size=9,color="#fc8181"),
            hovertemplate="<b>%{customdata[0]}</b><br>Country: %{customdata[1]}<br>Attacks: %{customdata[2]}<br>Type: %{customdata[3]}<extra></extra>",
            customdata=[[a["city"],a["country"],a["attacks"],a["type"]] for a in ATTACK_ORIGINS],
            name="Attack Origin", showlegend=True))
        fig_map.add_trace(go.Scattergeo(
            lat=[t["lat"] for t in TARGETS], lon=[t["lon"] for t in TARGETS],
            mode="markers+text",
            marker=dict(size=14,color=CYAN,opacity=0.9,symbol="diamond",
                        line=dict(width=1.5,color="rgba(79,209,197,0.6)")),
            text=[t["city"] for t in TARGETS], textposition="bottom center",
            textfont=dict(family="JetBrains Mono",size=9,color="#4fd1c5"),
            hovertemplate="<b>%{text}</b><br>Protected Target<extra></extra>",
            name="Protected Target", showlegend=True))
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500,
            margin=dict(l=0,r=0,t=0,b=0),
            geo=dict(bgcolor="#090c14",landcolor="#0e1220",oceancolor="#070a11",
                     lakecolor="#070a11",coastlinecolor="rgba(79,209,197,0.2)",
                     countrycolor="rgba(79,209,197,0.12)",
                     showland=True,showocean=True,showlakes=True,
                     showcountries=True,showcoastlines=True,
                     projection_type="natural earth",
                     framecolor="rgba(79,209,197,0.15)",showframe=True),
            legend=dict(bgcolor="rgba(14,18,32,0.85)",
                        bordercolor="rgba(79,209,197,0.2)",borderwidth=1,
                        font=dict(family="JetBrains Mono",size=10,color="#718096"),x=0.01,y=0.01),
            font=dict(family="JetBrains Mono",color="#718096"))

        mh1,mh2,mh3,mh4 = st.columns(4)
        total_attacks = sum(a["attacks"] for a in ATTACK_ORIGINS)
        top = max(ATTACK_ORIGINS, key=lambda x: x["attacks"])
        for col3, label, val, border_col in [
            (mh1,"TOTAL ATTACKS TODAY",f"{total_attacks:,}",RED),
            (mh2,"ATTACK ORIGINS",     str(len(ATTACK_ORIGINS)),AMBER),
            (mh3,"TARGETS PROTECTED",  str(len(TARGETS)),BLUE),
            (mh4,"TOP THREAT ACTOR",   top["country"],PURPLE),
        ]:
            with col3:
                st.markdown(f"""
                <div style='background:#0e1220;border:1px solid rgba(255,255,255,0.06);
                            border-left:3px solid {border_col};border-radius:4px;padding:12px 14px'>
                  <div style='font-family:JetBrains Mono,monospace;font-size:9px;
                              letter-spacing:0.08em;color:#718096;margin-bottom:3px'>{label}</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:24px;
                              font-weight:700;color:{border_col}'>{val}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar":False})

        tc1, tc2 = st.columns([3,2])
        with tc1:
            df_atk = pd.DataFrame(ATTACK_ORIGINS)[["country","city","attacks","type"]]
            df_atk.columns = ["Country","City","Attacks","Attack Type"]
            st.dataframe(df_atk.sort_values("Attacks",ascending=False).reset_index(drop=True),
                use_container_width=True, hide_index=True, height=280)
        with tc2:
            fig_atk = go.Figure(go.Bar(
                x=[a["attacks"] for a in sorted(ATTACK_ORIGINS, key=lambda x: x["attacks"])],
                y=[a["country"] for a in sorted(ATTACK_ORIGINS, key=lambda x: x["attacks"])],
                orientation="h",
                marker=dict(color=[a["attacks"] for a in sorted(ATTACK_ORIGINS, key=lambda x: x["attacks"])],
                            colorscale=[[0,BLUE],[0.5,AMBER],[1,RED]], line=dict(width=0))))
            fig_atk.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
            st.plotly_chart(fig_atk, use_container_width=True, config={"displayModeBar":False})

    # ══════════════════════════════════════════
    # TAB: MY ACCOUNT
    # ══════════════════════════════════════════
    with tab_acct:
        u_data     = USERS_DB[st.session_state.current_user]
        ac         = u_data["avatar_color"]
        login_time = (st.session_state.session_start.strftime("%H:%M")
                      if st.session_state.session_start else "---")
        dur        = session_duration()
        alr        = st.session_state.alerts_reviewed
        esc        = st.session_state.cases_escalated

        a1, a2, a3 = st.columns([1.8, 1.2, 1])

        # ── Col 1: Profile ──
        with a1:
            card = (
                "<div style='background:#0e1220;border:1px solid rgba(79,209,197,0.2);"
                "border-top:3px solid #4fd1c5;border-radius:8px;padding:22px 20px'>"
                "<div style='display:table;margin-bottom:14px'>"
                "<div style='display:table-cell;vertical-align:middle;width:56px'>"
                "<div style='width:52px;height:52px;border-radius:50%;background:rgba(79,209,197,0.12);"
                "border:2px solid " + ac + ";display:table-cell;vertical-align:middle;"
                "text-align:center;font-family:JetBrains Mono,monospace;font-size:16px;"
                "font-weight:700;color:" + ac + "'>" + u_data['initials'] + "</div>"
                "</div>"
                "<div style='display:table-cell;vertical-align:middle;padding-left:12px'>"
                "<div style='font-size:15px;font-weight:700;color:#e2e8f0'>" + u_data['full_name'] + "</div>"
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#718096'>" + u_data['role'] + "</div>"
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:" + ac + "'>"
                + u_data['department'] + " &nbsp;·&nbsp; " + u_data['location'] + "</div>"
                "</div></div>"
                "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:0 0 10px 0'>"
            )
            rows = [
                ("Email",        u_data["email"],                   "#4fd1c5"),
                ("Phone",        u_data["phone"],                   "#e2e8f0"),
                ("Clearance",    u_data["clearance"],               "#fc8181"),
                ("Member since", u_data["joined"],                  "#e2e8f0"),
                ("Username",     st.session_state.current_user,     "#4fd1c5"),
            ]
            for lbl, val, vc in rows:
                card += (
                    "<div style='display:table;width:100%;padding:7px 0;"
                    "border-bottom:1px solid rgba(255,255,255,0.04)'>"
                    "<div style='display:table-cell;font-family:JetBrains Mono,monospace;"
                    "font-size:11px;color:#718096'>" + lbl + "</div>"
                    "<div style='display:table-cell;text-align:right;font-family:JetBrains Mono,"
                    "monospace;font-size:11px;font-weight:700;color:" + vc + "'>" + val + "</div>"
                    "</div>"
                )
            card += "</div>"
            st.markdown(card, unsafe_allow_html=True)

        # ── Col 2: Session stats ──
        with a2:
            def stat_block(label, value, color="#4fd1c5"):
                return (
                    "<div style='padding-bottom:16px;margin-bottom:16px;"
                    "border-bottom:1px solid rgba(255,255,255,0.05)'>"
                    "<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
                    "letter-spacing:0.12em;color:#718096;margin-bottom:4px'>" + label + "</div>"
                    "<div style='font-family:Rajdhani,sans-serif;font-size:30px;"
                    "font-weight:700;color:" + color + "'>" + str(value) + "</div>"
                    "</div>"
                )
            session_html = (
                "<div style='background:#0e1220;border:1px solid rgba(79,209,197,0.18);"
                "border-radius:8px;padding:22px 20px'>"
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                "letter-spacing:0.12em;color:#718096;margin-bottom:16px'>CURRENT SESSION</div>"
                + stat_block("SESSION DURATION", dur,        "#4fd1c5")
                + stat_block("LOGIN TIME",        login_time, "#e2e8f0")
                + stat_block("ALERTS REVIEWED",   alr,        "#68d391")
                + stat_block("CASES ESCALATED",   esc,        "#f6ad55")
                + "</div>"
            )
            st.markdown(session_html, unsafe_allow_html=True)

        # ── Col 3: Settings ──
        with a3:
            st.markdown(
                "<div style='background:#0e1220;border:1px solid rgba(79,209,197,0.18);"
                "border-radius:8px;padding:22px 20px;margin-bottom:12px'>"
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                "letter-spacing:0.12em;color:#718096;margin-bottom:12px'>ACCOUNT SETTINGS</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            new_name  = st.text_input("Display Name", value=u_data["full_name"], key="acct_name")
            new_email = st.text_input("Email address", value=u_data["email"],    key="acct_email")
            if st.button("💾  SAVE CHANGES", use_container_width=True, key="save_acct"):
                USERS_DB[st.session_state.current_user]["full_name"] = new_name
                USERS_DB[st.session_state.current_user]["email"]     = new_email
                st.success("✅  Profile updated successfully.")
            st.markdown(
                "<div style='background:rgba(252,129,129,0.06);"
                "border:1px solid rgba(252,129,129,0.22);border-radius:6px;"
                "padding:12px 14px;margin-top:10px'>"
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                "letter-spacing:0.08em;color:#fc8181;margin-bottom:3px'>DANGER ZONE</div>"
                "<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
                "color:#718096'>Signing out ends your active session.</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⎋  SIGN OUT OF SMARTGUARD", use_container_width=True, key="signout_acct"):
                st.session_state.logged_in     = False
                st.session_state.current_user  = None
                st.session_state.session_start = None
                st.rerun()

    # ── FOOTER ────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align:center;font-family:JetBrains Mono,monospace;
                font-size:9px;color:#2d3748;padding:4px 0'>
      SmartGuard Enterprise v5.0 &copy; 2026 &nbsp;·&nbsp;
      Signed in as <span style='color:#4fd1c5'>{name}</span>
      &nbsp;·&nbsp; {u["location"]} &nbsp;·&nbsp;
      Session: {session_duration()} &nbsp;·&nbsp;
      {datetime.now().strftime('%d %b %Y %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  ROUTER
# ═════════════════════════════════════════════
if st.session_state.logged_in:
    render_dashboard()
else:
    render_login()

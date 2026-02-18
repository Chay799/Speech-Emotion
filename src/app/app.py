import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Indic SER", layout="wide")

# ========== CUSTOM CSS ==========
st.markdown("""
<style>

/* Hide streamlit header/footer */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main background */
.stApp {
    background: linear-gradient(120deg,#0f172a,#1e293b,#020617);
    color: white;
}

/* Hero title */
.hero {
    text-align:center;
    padding-top:30px;
}
.hero h1 {
    font-size:60px;
    color:#f97316;
}
.hero p {
    font-size:20px;
    color:#cbd5e1;
}

/* Glass card */
.card {
    background: rgba(255,255,255,0.05);
    padding:30px;
    border-radius:20px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}

/* Result emotion big */
.result {
    text-align:center;
    font-size:40px;
    color:#22c55e;
}

/* Confidence */
.conf {
    text-align:center;
    font-size:22px;
    color:#22c55e;
}

</style>
""", unsafe_allow_html=True)

# ========== HERO ==========
st.markdown("""
<div class='hero'>
<h1>🎤 Speech Emotion Recognition</h1>
<p>Detect emotions from Telugu & Kannada speech using Deep Learning</p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.write("")

# ========== TWO COLUMN LAYOUT ==========
col1, col2 = st.columns(2)

# ---------- LEFT : Upload ----------
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📤 Upload Audio")

    audio_file = st.file_uploader("Upload WAV", type=["wav"])

    if audio_file:
        st.audio(audio_file)

        if st.button("Analyze Emotion"):
            files = {"file": audio_file.getvalue()}
            res = requests.post(API_URL, files=files).json()

            st.session_state["emotion"] = res["emotion"]
            st.session_state["confidence"] = res["confidence"]

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT : Results ----------
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📊 Results")

    if "emotion" in st.session_state:
        emotion = st.session_state["emotion"]
        conf = st.session_state["confidence"]

        st.markdown(f"<div class='result'>{emotion}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='conf'>Confidence: {conf:.2f}</div>", unsafe_allow_html=True)

        st.write("")
        st.subheader("Probabilities")

        # Fake bars for UI (like demo)
        st.progress(conf if "NEUTRAL" in emotion else 0.3)
        st.write("NEUTRAL")

        st.progress(conf if "ANGRY" in emotion else 0.3)
        st.write("Angry")

        st.progress(conf if "SAD" in emotion else 0.3)
        st.write("Sad")

    else:
        st.write("Upload audio to see prediction")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.write("")

# ========== FOOTER STATS ==========
st.markdown("""
<div class='card'>
<h2>📊 Model Stats</h2>
Telugu + Kannada | CNN + Attention | 4 sec audio | 3 emotions
</div>
""", unsafe_allow_html=True)

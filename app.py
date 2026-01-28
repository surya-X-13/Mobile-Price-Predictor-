import streamlit as st
import pandas as pd
import pickle
import numpy as np

with open("final_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("best_logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURES = [
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
    'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
    'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
    'touch_screen', 'wifi'
]

PRICE_MAP = ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"]

st.set_page_config(
    page_title="📱 Mobile Price Predictor Pro",
    page_icon="📱",
    layout="wide"
)


theme = st.toggle("🌙 Dark Mode", value=True)

bg = "#0f2027" if theme else "#f4f6f9"
text = "white" if theme else "#111"

st.markdown(f"""
<style>
body {{
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: {text};
}}
.card {{
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(15px);
    padding: 25px;
    border-radius: 25px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.4);
    margin-bottom: 25px;
}}
.title {{
    text-align:center;
    font-size:3rem;
    font-weight:900;
}}
.subtitle {{
    text-align:center;
    opacity:0.9;
    margin-bottom:30px;
}}
.result {{
    background: linear-gradient(135deg, #ff512f, #dd2476);
    padding: 35px;
    border-radius: 30px;
    text-align:center;
    font-size:2.2rem;
    font-weight:800;
    color:white;
}}
.stButton>button {{
    width:100%;
    padding:16px;
    border-radius:35px;
    font-size:1.4rem;
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color:white;
    border:none;
}}
footer {{
    text-align:center;
    opacity:0.7;
    margin-top:30px;
}}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="title">📱 Mobile Price Predictor Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered smartphone price intelligence</div>', unsafe_allow_html=True)


st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔧 Hardware Specs")

c1, c2, c3 = st.columns(3)

with c1:
    battery_power = st.number_input("🔋 Battery Power", 0, 5000)
    ram = st.number_input("🧠 RAM (MB)", 0, 8000)
    int_memory = st.number_input("💾 Storage (GB)", 0, 128)

with c2:
    clock_speed = st.number_input("⏱ CPU Speed (GHz)", 0.0, 3.0, 0.1)
    n_cores = st.number_input("🖥 CPU Cores", 1, 8)
    mobile_wt = st.number_input("⚖ Weight (g)", 0, 300)

with c3:
    pc = st.number_input("📸 Camera (MP)", 0, 100)
    px_height = st.number_input("🖼 Pixel Height", 0, 3000)
    px_width = st.number_input("🖼 Pixel Width", 0, 3000)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📡 Connectivity")

d1, d2, d3, d4 = st.columns(4)

with d1:
    blue = st.selectbox("Bluetooth", ["Yes", "No"])
    dual_sim = st.selectbox("Dual SIM", ["Yes", "No"])

with d2:
    four_g = st.selectbox("4G", ["Yes", "No"])
    three_g = st.selectbox("3G", ["Yes", "No"])

with d3:
    touch_screen = st.selectbox("Touch Screen", ["Yes", "No"])
    wifi = st.selectbox("WiFi", ["Yes", "No"])

with d4:
    fc = st.selectbox("NFC", ["Yes", "No"])

st.markdown('</div>', unsafe_allow_html=True)


yn = lambda x: 1 if x == "Yes" else 0

input_df = pd.DataFrame([{
    'battery_power': battery_power,
    'blue': yn(blue),
    'clock_speed': clock_speed,
    'dual_sim': yn(dual_sim),
    'fc': yn(fc),
    'four_g': yn(four_g),
    'int_memory': int_memory,
    'm_dep': 0.8,
    'mobile_wt': mobile_wt,
    'n_cores': n_cores,
    'pc': pc,
    'px_height': px_height,
    'px_width': px_width,
    'ram': ram,
    'sc_h': 10,
    'sc_w': 5,
    'talk_time': 20,
    'three_g': yn(three_g),
    'touch_screen': yn(touch_screen),
    'wifi': yn(wifi)
}], columns=FEATURES)


if st.button("🔮 Predict Price"):
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)[0]
    probs = model.predict_proba(scaled)[0]

    st.markdown(f"""
    <div class="result">
        {PRICE_MAP[pred]}
        <br>
        <span style="font-size:1.2rem;">Confidence Breakdown</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Model Confidence")
    for i, p in enumerate(probs):
        st.progress(float(p), text=f"{PRICE_MAP[i]} : {p*100:.2f}%")


st.markdown("<footer>🚀 Built with ML & ❤️ by My love Suryadip</footer>", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go # เพิ่มตัวนี้เพื่อทำกราฟ Radar สวยๆ

# --- [1. CONFIGURATION] ---
st.set_page_config(page_title="SportMatch AI Expert", layout="wide", page_icon="🏆")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT-CWhytgN4VrGjp82zbtzbjTrw0HtIqcUtOUuyt8HVwiJbwB9_gRBPt5MRlGUe6AGILR9W2-y8_Inf/pub?output=csv"
features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']

STATIC_DATA = {
    'Sport': ['วิ่ง', 'ว่ายน้ำ', 'โยคะ', 'ฟุตบอล', 'แบดมินตัน', 'มวยสากล', 'จักรยาน', 'ยกน้ำหนัก', 'เทนนิส', 'ปีนหน้าผา', 'ยิงธนู', 'บาสเกตบอล', 'หมากรุก'],
    'Intensity': [8, 7, 3, 9, 7, 10, 6, 8, 8, 7, 2, 9, 1],
    'Social': [1, 1, 2, 10, 8, 2, 4, 1, 6, 5, 2, 10, 1],
    'Budget': [2, 4, 3, 3, 5, 4, 8, 5, 9, 7, 8, 4, 1],
    'Flexibility': [3, 8, 10, 5, 7, 6, 3, 2, 6, 8, 4, 6, 1],
    'Strength': [4, 6, 5, 6, 4, 8, 5, 10, 6, 9, 6, 7, 1]
}

# --- [2. FUNCTIONS] ---
@st.cache_data(ttl=1)
def load_data(use_google=True):
    if not use_google: return pd.DataFrame(STATIC_DATA)
    try:
        df = pd.read_csv(SHEET_URL).iloc[:, 1:7].dropna(how='all')
        df.columns = ['Sport', 'Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(5)
        return df
    except: return pd.DataFrame(STATIC_DATA)

# ฟังก์ชันสร้างกราฟ Radar ให้สวยงาม
def create_radar_chart(row):
    categories = ['Cardio', 'Social', 'Budget', 'Flex', 'Strength']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[row['Intensity'], row['Social'], row['Budget'], row['Flexibility'], row['Strength']],
        theta=categories,
        fill='toself',
        name=row['Sport'],
        line_color='#4F8BF9'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=250
    )
    return fig

# ฟังก์ชันเหตุผลแบบหลากหลาย (ไม่ซ้ำซาก)
def get_diverse_reason(row, old_row, user_req, is_newbie):
    reasons = []
    # วิเคราะห์จุดเด่นที่แมตช์กับ User
    if row['Intensity'] >= 7 and user_req[0] >= 7:
        reasons.append(f"เป็นตัวเลือกที่ยอดเยี่ยมสำหรับการรีดสมรรถนะหัวใจ (Cardio สูงถึง {row['Intensity']}/10)")
    if row['Budget'] <= 3:
        reasons.append("ประหยัดงบประมาณและเริ่มเล่นได้ทันทีโดยไม่ต้องลงทุนสูง")
    if row['Social'] >= 8:
        reasons.append("เน้นมิตรภาพและการทำงานเป็นทีมที่เข้มข้น")
    
    # กรณีอ้างอิงพื้นฐานเดิม
    if not is_newbie and old_row is not None:
        if row['Intensity'] > old_row['Intensity']:
            reasons.append(f"ท้าทายความอึดได้มากกว่า {old_row['Sport']} ที่คุณเคยเล่น")
        reasons.append(f"ช่วยเสริมสร้างทักษะใหม่ที่เข้ามาเติมเต็มพื้นฐานเดิมจาก {old_row['Sport']}")
    elif is_newbie:
        reasons.append("โครงสร้างกีฬานี้ถูกออกแบบมาให้มือใหม่เข้าถึงง่ายและลดโอกาสการบาดเจ็บ")

    # สรุปประโยคให้ดูเป็นธรรมชาติ
    if not reasons: return "คุณสมบัติโดยรวมของกีฬานี้มีความสมดุลและสอดคล้องกับค่าที่คุณเลือกใน Slider มากที่สุด"
    return " อีกทั้งยัง ".join(reasons[:2])

# --- [3. UI INTERFACE] ---
with st.sidebar:
    st.header("⚙️ ตั้งค่าแหล่งข้อมูล")
    source = st.radio("เลือกแหล่งข้อมูลที่จะใช้:", ["Google Form (Real-time)", "ข้อมูลระบบ (Static)"])
    df = load_data(use_google=(source == "Google Form (Real-time)"))
    st.divider()
    st.info("💡 ระบบประมวลผลผ่านโมเดล Vector Distance")

st.title("🏆 SportMatch AI Expert")
st.caption("ระบบวิเคราะห์กีฬาอัจฉริยะด้วย Machine Learning")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("👤 ข้อมูลผู้ใช้งาน")
    experience = st.radio("พื้นฐานกีฬา:", ["ไม่มีพื้นฐาน", "เคยเล่นกีฬาบางชนิดมาก่อน"])
    old_sport_row = None
    bonus_vector = np.zeros(5)
    if experience == "เคยเล่นกีฬาบางชนิดมาก่อน":
        selected_old = st.selectbox("เลือกกีฬาที่คุณเล่นอยู่ในปัจจุบัน:", df['Sport'].unique())
        old_sport_row = df[df['Sport'] == selected_old].iloc[0]
        bonus_vector = old_sport_row[features].values
        st.success(f"อ้างอิงฐานทักษะ: {selected_old}")

with col_right:
    st.subheader("🎯 เป้าหมายที่ต้องการ")
    c1, c2 = st.columns(2)
    with c1:
        u_int = st.select_slider("1. ความเหนื่อย (Cardio)", options=range(1, 11), value=5)
        u_soc = st.select_slider("2. การเข้าสังคม (Social)", options=range(1, 11), value=5)
        u_bud = st.select_slider("3. งบประมาณ (Budget)", options=range(1, 11), value=5)
    with c2:
        u_flex = st.select_slider("4. ความยืดหยุ่น", options=range(1, 11), value=5)
        u_str = st.select_slider("5. พละกำลัง", options=range(1, 11), value=5)
    run_btn = st.button("🚀 เริ่มการวิเคราะห์", use_container_width=True)

if run_btn:
    user_req = np.array([u_int, u_soc, u_bud, u_flex, u_str])
    is_newbie = (experience == "ไม่มีพื้นฐาน")
    final_vector = (user_req * 0.7) + (bonus_vector * 0.3) if not is_newbie else user_req
    sim = cosine_similarity([final_vector], df[features])
    df['Score'] = sim[0]
    
    processed_df = df.copy()
    if not is_newbie: processed_df = processed_df[processed_df['Sport'] != selected_old]
    recs = processed_df.sort_values(by='Score', ascending=False).head(3)
    
    st.divider()
    st.header("🔥 รายการที่แนะนำจากฐานข้อมูล")
    
    for i, (idx, row) in enumerate(recs.iterrows(), 1):
        with st.container():
            c_rank, c_info, c_plot = st.columns([0.8, 3.5, 2.5])
            with c_rank:
                st.markdown(f"<h1 style='color:#4F8BF9; font-size:60px;'>#{i}</h1>", unsafe_allow_html=True)
            with c_info:
                st.markdown(f"### **{row['Sport']}** (Match: {round(row['Score']*100, 1)}%)")
                reason = get_diverse_reason(row, old_sport_row, user_req, is_newbie)
                # ปรับ UI เหตุผลให้สวยขึ้น
                st.markdown(f"""
                <div style="background-color: #f0f7ff; padding: 15px; border-radius: 10px; border-left: 5px solid #4F8BF9;">
                    <strong>AI วิเคราะห์เหตุผล:</strong> {reason}
                </div>
                """, unsafe_allow_html=True)
            with c_plot:
                # แสดงกราฟ Radar แทนกราฟแท่ง
                st.plotly_chart(create_radar_chart(row), use_container_width=True, config={'displayModeBar': False})
            st.markdown("---")

st.divider()
st.subheader(f"📊 ฐานข้อมูลที่ใช้ ({source})")
st.dataframe(df.drop(columns=['Score'], errors='ignore'), use_container_width=True)

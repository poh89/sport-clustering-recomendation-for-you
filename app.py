import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- [CONFIGURATION] ---
st.set_page_config(page_title="SportMatch AI Pro+", layout="wide", page_icon="🏆")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT-CWhytgN4VrGjp82zbtzbjTrw0HtIqcUtOUuyt8HVwiJbwB9_gRBPt5MRlGUe6AGILR9W2-y8_Inf/pub?output=csv"

STATIC_DATA = {
    'Sport': ['วิ่ง', 'ว่ายน้ำ', 'โยคะ', 'ฟุตบอล', 'แบดมินตัน', 'มวยสากล', 'จักรยาน', 'ยกน้ำหนัก', 'เทนนิส', 'ปีนหน้าผา'],
    'Intensity': [8, 7, 3, 9, 7, 10, 6, 8, 8, 7],
    'Social': [1, 1, 2, 10, 8, 2, 4, 1, 6, 5],
    'Budget': [2, 4, 3, 3, 5, 4, 8, 5, 9, 7],
    'Flexibility': [3, 8, 10, 5, 7, 6, 3, 2, 6, 8],
    'Strength': [4, 6, 5, 6, 4, 8, 5, 10, 6, 9]
}

# --- [LOGIC FUNCTIONS] ---
@st.cache_data(ttl=1)
def load_data(use_google=True):
    if not use_google: return pd.DataFrame(STATIC_DATA)
    try:
        df = pd.read_csv(SHEET_URL).iloc[:, 1:7].dropna()
        df.columns = ['Sport', 'Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']
        for col in ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(5)
        return df
    except: return pd.DataFrame(STATIC_DATA)

# --- [UI SETUP] ---
st.title("🏆 SportMatch AI Pro+")
st.caption("ระบบวิเคราะห์กีฬาแบบ Hybrid: ผสานความต้องการใหม่กับพื้นฐานทักษะเดิม")

with st.sidebar:
    st.header("⚙️ แหล่งข้อมูล")
    source = st.radio("ใช้ฐานข้อมูลจาก:", ["Google Form (Real-time)", "ข้อมูลระบบ (Static)"])
    st.divider()
    st.markdown("### 🧠 AI Logic\nระบบจะคำนวณ **Weighted Vector** ระหว่างความต้องการปัจจุบันและพื้นฐานเดิมของคุณ")

df = load_data(use_google=(source == "Google Form (Real-time)"))
features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']

tab1, tab2 = st.tabs(["🎯 วิเคราะห์ระดับมืออาชีพ", "📊 ดูฐานข้อมูล"])

with tab1:
    col_a, col_b = st.columns([1.2, 2])
    
    with col_a:
        st.subheader("👤 ข้อมูลพื้นฐาน")
        experience = st.radio("คุณมีพื้นฐานกีฬามาก่อนหรือไม่?", ["ไม่มีพื้นฐาน (มือใหม่)", "เคยเล่นกีฬาบางชนิดมาก่อน"])
        
        bonus_vector = np.zeros(5)
        if experience == "เคยเล่นกีฬาบางชนิดมาก่อน":
            old_sport = st.selectbox("เลือกกีฬาที่ท่านชำนาญ/เล่นบ่อยที่สุด:", df['Sport'].unique())
            # ดึงค่า Vector ของกีฬาเก่ามาช่วยคำนวณ
            bonus_vector = df[df['Sport'] == old_sport][features].values[0]
            st.write(f"💡 AI จะนำทักษะจาก **{old_sport}** มาเป็นฐานการคำนวณ")

    with col_b:
        st.subheader("🎯 ความต้องการปัจจุบัน")
        c1, c2 = st.columns(2)
        with c1:
            u_int = st.select_slider("ต้องการความเหนื่อยแค่ไหน?", options=range(1, 11), value=5)
            u_soc = st.select_slider("ชอบสังคม/เล่นเป็นทีมไหม?", options=range(1, 11), value=5)
            u_bud = st.select_slider("งบประมาณ (1=ประหยัด)", options=range(1, 11), value=5)
        with c2:
            u_flex = st.select_slider("ความยืดหยุ่นที่ต้องการ", options=range(1, 11), value=5)
            u_str = st.select_slider("ความแรง/พละกำลัง", options=range(1, 11), value=5)
        
        run_btn = st.button("✨ วิเคราะห์หาผลลัพธ์ที่แม่นยำที่สุด", use_container_width=True)

    if run_btn:
        # การคำนวณแบบ Hybrid (ต้องการ 70% + พื้นฐานเดิม 30%)
        current_req = np.array([u_int, u_soc, u_bud, u_flex, u_str])
        if experience == "เคยเล่นกีฬาบางชนิดมาก่อน":
            final_vector = (current_req * 0.7) + (bonus_vector * 0.3)
        else:
            final_vector = current_req
            
        sim = cosine_similarity([final_vector], df[features])
        df['Score'] = sim[0]
        recs = df.sort_values(by='Score', ascending=False).head(3)
        
        st.divider()
        st.subheader("🔥 กีฬาที่แนะนำสำหรับคุณ")
        cols = st.columns(3)
        for i, (idx, row) in enumerate(recs.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 15px; border-top: 5px solid #4F8BF9; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h2 style="color: #1E3A8A; margin-bottom:0;">{row['Sport']}</h2>
                    <p style="color: #6B7280;">Match Score: {round(row['Score']*100, 1)}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # กราฟเปรียบเทียบคุณสมบัติ
                chart_data = pd.DataFrame({
                    'Attribute': ['Cardio', 'Social', 'Budget', 'Flex', 'Str'],
                    'Score': [row['Intensity'], row['Social'], row['Budget'], row['Flexibility'], row['Strength']]
                }).set_index('Attribute')
                st.bar_chart(chart_data, height=150)

with tab2:
    st.subheader("📊 ข้อมูลกีฬาในระบบปัจจุบัน")
    st.dataframe(df.drop(columns=['Score'], errors='ignore'), use_container_width=True)

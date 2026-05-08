import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- [CONFIGURATION & ASSETS] ---
st.set_page_config(page_title="SportMatch AI Pro", layout="wide", page_icon="🏆")

# ลิงก์ CSV จาก Google Sheet ของคุณ
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT-CWhytgN4VrGjp82zbtzbjTrw0HtIqcUtOUuyt8HVwiJbwB9_gRBPt5MRlGUe6AGILR9W2-y8_Inf/pub?output=csv"

# ข้อมูลสำรอง (Fallback Data) กรณีดึงจาก Google ไม่ได้ หรือข้อมูลยังว่างอยู่
STATIC_DATA = {
    'Sport': ['วิ่ง', 'ว่ายน้ำ', 'โยคะ', 'ฟุตบอล', 'แบดมินตัน', 'มวยสากล', 'จักรยาน', 'ยกน้ำหนัก'],
    'Intensity': [8, 7, 3, 9, 7, 10, 6, 8],
    'Social': [1, 1, 2, 10, 8, 2, 4, 1],
    'Budget': [2, 4, 3, 3, 5, 4, 8, 5],
    'Flexibility': [3, 8, 10, 5, 7, 6, 3, 2],
    'Strength': [4, 6, 5, 6, 4, 8, 5, 10]
}

# --- [LOGIC FUNCTIONS] ---
@st.cache_data(ttl=1)
def load_data(use_google=True):
    if not use_google:
        return pd.DataFrame(STATIC_DATA)
    try:
        df = pd.read_csv(SHEET_URL)
        # ทำความสะอาดข้อมูล: เลือก 7 คอลัมน์แรก และลบแถวที่ว่าง
        df = df.iloc[:, :7].dropna()
        df.columns = ['Timestamp', 'Sport', 'Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']
        # แปลงคอลัมน์คะแนนให้เป็นตัวเลขจริงๆ (ป้องกันคนกรอกตัวอักษร)
        for col in ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(5)
        return df
    except Exception as e:
        st.sidebar.warning(f"เชื่อมต่อ Google Sheet ไม่ได้: ใช้ข้อมูลระบบแทน")
        return pd.DataFrame(STATIC_DATA)

# --- [UI COMPONENTS] ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #4F8BF9; color: white; }
    .sport-card { background-color: white; padding: 20px; border-radius: 15px; border-left: 5px solid #4F8BF9; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("🏆 SportMatch AI Pro")
st.caption("ระบบวิเคราะห์และแนะนำกีฬาอัจฉริยะด้วย Machine Learning")

# แถบเมนูด้านข้าง
with st.sidebar:
    st.header("⚙️ การตั้งค่า")
    source = st.radio("แหล่งข้อมูลที่ใช้:", ["Google Form (Real-time)", "ข้อมูลมาตรฐาน (Static)"])
    st.divider()
    st.info("💡 คำแนะนำ: ระบบจะคำนวณความเหมาะสมจากระยะห่างของ Vector (Cosine Similarity)")

# โหลดข้อมูล
is_google = (source == "Google Form (Real-time)")
df = load_data(use_google=is_google)
features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']

# --- [MAIN INTERFACE] ---
tab1, tab2, tab3 = st.tabs(["🎯 แนะนำกีฬา", "🔄 เปลี่ยนสายกีฬา", "📊 ฐานข้อมูล"])

with tab1:
    st.subheader("ค้นหากีฬาที่ตรงใจคุณ")
    c1, c2 = st.columns([1, 1])
    with c1:
        u_int = st.select_slider("ความเหนื่อย / Cardio", options=range(1, 11), value=5)
        u_soc = st.select_slider("การเข้าสังคม / ทีม", options=range(1, 11), value=5)
        u_bud = st.select_slider("งบประมาณ (1=ประหยัด)", options=range(1, 11), value=5)
    with c2:
        u_flex = st.select_slider("ความยืดหยุ่นร่างกาย", options=range(1, 11), value=5)
        u_str = st.select_slider("พละกำลัง / กล้ามเนื้อ", options=range(1, 11), value=5)
        st.write("---")
        run_btn = st.button("✨ วิเคราะห์หาผลลัพธ์")

    if run_btn:
        user_vector = np.array([[u_int, u_soc, u_bud, u_flex, u_str]])
        sim = cosine_similarity(user_vector, df[features])
        df['Score'] = sim[0]
        recs = df.sort_values(by='Score', ascending=False).head(3)
        
        st.balloons()
        cols = st.columns(3)
        for i, (idx, row) in enumerate(recs.iterrows()):
            with cols[i]:
                st.markdown(f"""<div class="sport-card">
                    <h3>{row['Sport']}</h3>
                    <p>ความเหมาะสม: <b>{round(row['Score']*100, 1)}%</b></p>
                </div>""", unsafe_allow_html=True)
                # เพิ่มกราฟแท่งเล็กๆ เปรียบเทียบ
                chart_data = pd.DataFrame({
                    'Feature': ['Cardio', 'Social', 'Budget', 'Flex', 'Str'],
                    'Score': [row['Intensity'], row['Social'], row['Budget'], row['Flexibility'], row['Strength']]
                })
                st.bar_chart(chart_data.set_index('Feature'), height=150)

with tab2:
    st.subheader("เบื่อกีฬาเดิมหรือยัง? ลองหาอย่างอื่นที่ทักษะใกล้เคียงกัน")
    all_sports = df['Sport'].unique()
    current = st.selectbox("เลือกกีฬาที่คุณเล่นอยู่ในปัจจุบัน:", all_sports)
    
    if st.button("🚀 ค้นหาความท้าทายใหม่"):
        target_vec = df[df['Sport'] == current][features].values
        others = df[df['Sport'] != current].copy()
        
        if not others.empty:
            sim = cosine_similarity(target_vec, others[features])
            others['Score'] = sim[0]
            top_new = others.sort_values(by='Score', ascending=False).head(3)
            
            st.write(f"เพราะคุณถนัด **{current}** คุณจะเรียนรู้กีฬาเหล่านี้ได้เร็วมาก:")
            cols = st.columns(3)
            for i, (idx, row) in enumerate(top_new.iterrows()):
                with cols[i]:
                    st.success(f"**{row['Sport']}**")
                    st.write(f"ความคล้ายคลึง: {round(row['Score']*100, 1)}%")
        else:
            st.error("ข้อมูลไม่เพียงพอที่จะเปรียบเทียบ")

with tab3:
    st.subheader("ข้อมูลทั้งหมดในระบบ")
    st.dataframe(df.drop(columns=['Score'], errors='ignore'), use_container_width=True)
    st.download_button("📩 Download Data (CSV)", df.to_csv(index=False), "sports_data.csv")

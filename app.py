import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- [CONFIGURATION] ---
st.set_page_config(page_title="SportMatch AI Expert", layout="wide", page_icon="🏆")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT-CWhytgN4VrGjp82zbtzbjTrw0HtIqcUtOUuyt8HVwiJbwB9_gRBPt5MRlGUe6AGILR9W2-y8_Inf/pub?output=csv"

STATIC_DATA = {
    'Sport': ['วิ่ง', 'ว่ายน้ำ', 'โยคะ', 'ฟุตบอล', 'แบดมินตัน', 'มวยสากล', 'จักรยาน', 'ยกน้ำหนัก', 'เทนนิส', 'ปีนหน้าผา', 'ยิงธนู'],
    'Intensity': [8, 7, 3, 9, 7, 10, 6, 8, 8, 7, 2],
    'Social': [1, 1, 2, 10, 8, 2, 4, 1, 6, 5, 2],
    'Budget': [2, 4, 3, 3, 5, 4, 8, 5, 9, 7, 8],
    'Flexibility': [3, 8, 10, 5, 7, 6, 3, 2, 6, 8, 4],
    'Strength': [4, 6, 5, 6, 4, 8, 5, 10, 6, 9, 6]
}

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

# ฟังก์ชันสร้างคำอธิบายอัตโนมัติ
def get_reason(row, user_req):
    reasons = []
    if row['Intensity'] >= 7 and user_req[0] >= 7: reasons.append("เน้นการเผาผลาญแคลอรี่สูงตามที่คุณต้องการ")
    if row['Social'] >= 7 and user_req[1] >= 7: reasons.append("ตอบโจทย์สายสังคมและเน้นการทำงานเป็นทีม")
    if row['Budget'] <= 4 and user_req[2] <= 4: reasons.append("เป็นกีฬาที่ประหยัดงบประมาณและเริ่มได้ง่าย")
    if row['Flexibility'] >= 7 and user_req[3] >= 7: reasons.append("ช่วยเสริมสร้างความยืดหยุ่นและสมาธิ")
    if row['Strength'] >= 7 and user_req[4] >= 7: reasons.append("เน้นการสร้างกล้ามเนื้อและพละกำลังที่แข็งแกร่ง")
    
    return " และ ".join(reasons) if reasons else "มีสมดุลของคุณสมบัติใกล้เคียงกับสไตล์ที่คุณเลือกมากที่สุด"

st.title("🏆 SportMatch AI Expert")
st.caption("ระบบวิเคราะห์และแนะนำกีฬาชนิดถัดไป (Next-Gen Sports Recommendation)")

with st.sidebar:
    st.header("⚙️ แหล่งข้อมูล")
    source = st.radio("ใช้ฐานข้อมูลจาก:", ["Google Form (Real-time)", "ข้อมูลระบบ (Static)"])
    df = load_data(use_google=(source == "Google Form (Real-time)"))

tab1, tab2 = st.tabs(["🎯 วิเคราะห์กีฬาชนิดถัดไป", "📊 ตรวจสอบฐานข้อมูล"])

with tab1:
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("👤 ประวัติของคุณ")
        experience = st.radio("พื้นฐานกีฬา:", ["มือใหม่", "เคยเล่นกีฬาบางชนิดมาก่อน"])
        old_sport = None
        bonus_vector = np.zeros(5)
        if experience == "เคยเล่นกีฬาบางชนิดมาก่อน":
            old_sport = st.selectbox("เลือกกีฬาที่คุณเล่นอยู่ในปัจจุบัน:", df['Sport'].unique())
            bonus_vector = df[df['Sport'] == old_sport][features].values[0]

    with col_b:
        st.subheader("🎯 เป้าหมายใหม่")
        features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']
        c1, c2 = st.columns(2)
        with c1:
            u_int = st.select_slider("Cardio", options=range(1, 11), value=5)
            u_soc = st.select_slider("Social", options=range(1, 11), value=5)
            u_bud = st.select_slider("Budget", options=range(1, 11), value=5)
        with c2:
            u_flex = st.select_slider("Flexibility", options=range(1, 11), value=5)
            u_str = st.select_slider("Strength", options=range(1, 11), value=5)
        run_btn = st.button("🚀 วิเคราะห์กีฬาอันดับถัดไปของคุณ", use_container_width=True)

    if run_btn:
        user_req = np.array([u_int, u_soc, u_bud, u_flex, u_str])
        final_vector = (user_req * 0.7) + (bonus_vector * 0.3) if old_sport else user_req
        
        sim = cosine_similarity([final_vector], df[features])
        df['Score'] = sim[0]
        
        # --- 💡 จุดแก้ไข: ตัดกีฬาเดิมออกจากการแนะนำ ---
        recs_df = df.copy()
        if old_sport:
            recs_df = recs_df[recs_df['Sport'] != old_sport]
            
        recs = recs_df.sort_values(by='Score', ascending=False).head(3)
        
        st.divider()
        st.subheader(f"🔥 กีฬาที่แนะนำเป็นอันดับถัดไป (ต่อจาก {old_sport if old_sport else 'มือใหม่'})")
        
        for i, (idx, row) in enumerate(recs.iterrows(), 1):
            with st.container():
                c_rank, c_info = st.columns([1, 5])
                with c_rank:
                    st.markdown(f"<h1 style='text-align:center; color:#4F8BF9;'>#{i}</h1>", unsafe_allow_html=True)
                with c_info:
                    st.markdown(f"### {row['Sport']} (ความแม่นยำ {round(row['Score']*100, 1)}%)")
                    reason_text = get_reason(row, user_req)
                    st.info(f"**เหตุผลที่แนะนำ:** {reason_text}")
                    
                    # กราฟเปรียบเทียบขนาดเล็ก
                    chart_data = pd.DataFrame({
                        'Attribute': ['Cardio', 'Social', 'Budget', 'Flex', 'Str'],
                        'Score': [row['Intensity'], row['Social'], row['Budget'], row['Flexibility'], row['Strength']]
                    }).set_index('Attribute')
                    st.bar_chart(chart_data.T, height=150)
                st.markdown("---")

with tab2:
    st.dataframe(df.drop(columns=['Score'], errors='ignore'), use_container_width=True)

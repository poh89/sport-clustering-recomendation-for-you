import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- [1. CONFIGURATION & SETUP] ---
st.set_page_config(page_title="SportMatch AI Professional", layout="wide", page_icon="🏆")

# ลิงก์เชื่อมต่อ Google Sheet (CSV)
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT-CWhytgN4VrGjp82zbtzbjTrw0HtIqcUtOUuyt8HVwiJbwB9_gRBPt5MRlGUe6AGILR9W2-y8_Inf/pub?output=csv"
features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']

# ข้อมูลสำรอง (Static Data) กันระบบล่ม
STATIC_DATA = {
    'Sport': ['วิ่ง', 'ว่ายน้ำ', 'โยคะ', 'ฟุตบอล', 'แบดมินตัน', 'มวยสากล', 'จักรยาน', 'ยกน้ำหนัก', 'เทนนิส', 'ปีนหน้าผา', 'ยิงธนู'],
    'Intensity': [8, 7, 3, 9, 7, 10, 6, 8, 8, 7, 2],
    'Social': [1, 1, 2, 10, 8, 2, 4, 1, 6, 5, 2],
    'Budget': [2, 4, 3, 3, 5, 4, 8, 5, 9, 7, 8],
    'Flexibility': [3, 8, 10, 5, 7, 6, 3, 2, 6, 8, 4],
    'Strength': [4, 6, 5, 6, 4, 8, 5, 10, 6, 9, 6]
}

# --- [2. FUNCTIONS: DATA & REASONING] ---
@st.cache_data(ttl=1)
def load_data():
    try:
        df = pd.read_csv(SHEET_URL).iloc[:, 1:7].dropna(how='all')
        df.columns = ['Sport', 'Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(5)
        return df if not df.empty else pd.DataFrame(STATIC_DATA)
    except:
        return pd.DataFrame(STATIC_DATA)

def get_expert_reason(new_row, old_row, user_req, mode):
    reasons = []
    
    # วิเคราะห์ตาม Slider ของ User
    if user_req[0] >= 7: reasons.append(f"เน้นการฝึกหัวใจ (Cardio) ระดับ {user_req[0]} ตามที่คุณต้องการ")
    if user_req[1] <= 3: reasons.append("ให้พื้นที่ส่วนตัวในการฝึกซ้อมสูง")
    if user_req[2] <= 4: reasons.append(f"ใช้งบประมาณน้อย (ระดับ {new_row['Budget']}) เริ่มได้ทันที")
    
    # กรณีเป็นคนมีพื้นฐาน: เปรียบเทียบกับกีฬาเดิม
    if mode == "Experienced" and old_row is not None:
        if new_row['Intensity'] > old_row['Intensity']:
            reasons.append(f"ท้าทายความอึดได้มากกว่า {old_row['Sport']} เดิมที่คุณเล่นอยู่")
        if new_row['Social'] > old_row['Social']:
            reasons.append(f"เพิ่มทักษะการทำงานเป็นทีมได้ดีกว่า {old_row['Sport']}")
        reasons.append(f"ช่วยเสริมสร้างกล้ามเนื้อในส่วนที่กีฬา {old_row['Sport']} อาจจะยังเข้าไม่ถึง")
    
    # กรณีมือใหม่
    elif mode == "Newbie":
        reasons.append(f"เป็นจุดเริ่มต้นที่ปลอดภัยและสร้างพื้นฐานร่างกายที่ดีสำหรับมือใหม่")

    return " อีกทั้งยัง ".join(reasons[:3])

# --- [3. UI INTERFACE] ---
st.title("🏆 SportMatch AI Professional")
st.caption("ระบบวิเคราะห์และแนะนำกีฬาอัจฉริยะด้วย Machine Learning (Hybrid Logic)")

df = load_data()

# ขั้นตอนที่ 1: เลือกสถานะ
st.subheader("👤 1. ระบุสถานะผู้ใช้งาน")
user_choice = st.radio("คุณคือกลุ่มใด:", ["มือใหม่ (ไม่มีพื้นฐาน)", "มีพื้นฐาน (เล่นกีฬาอยู่แล้ว)"], horizontal=True)
mode = "Newbie" if "มือใหม่" in user_choice else "Experienced"

st.divider()

col_in, col_out = st.columns([1, 2])

with col_in:
    st.subheader("🎯 2. กำหนดเป้าหมาย (Slider)")
    old_sport_row = None
    if mode == "Experienced":
        selected_old = st.selectbox("กีฬาที่คุณเล่นปัจจุบัน:", df['Sport'].unique())
        old_sport_row = df[df['Sport'] == selected_old].iloc[0]
        st.success(f"อ้างอิงทักษะเดิมจาก: {selected_old}")

    # Slider 5 ปัจจัยหลัก
    u_int = st.select_slider("ความเหนื่อย (Cardio)", options=range(1, 11), value=5)
    u_soc = st.select_slider("การเข้าสังคม (Social)", options=range(1, 11), value=5)
    u_bud = st.select_slider("งบประมาณ (Budget)", options=range(1, 11), value=5)
    u_flex = st.select_slider("ความยืดหยุ่น (Flex)", options=range(1, 11), value=5)
    u_str = st.select_slider("พละกำลัง (Strength)", options=range(1, 11), value=5)
    
    run_btn = st.button("🚀 เริ่มวิเคราะห์ผล", use_container_width=True)

with col_out:
    if run_btn:
        user_req = np.array([u_int, u_soc, u_bud, u_flex, u_str])
        bonus_vector = old_sport_row[features].values if old_sport_row is not None else np.zeros(5)
        
        # คำนวณความคล้ายคลึง (Hybrid: ทักษะเดิม 30% + ความต้องการใหม่ 70%)
        final_vector = (user_req * 0.7) + (bonus_vector * 0.3) if mode == "Experienced" else user_req
        
        sim = cosine_similarity([final_vector], df[features])
        df['Score'] = sim[0]
        
        # กรองข้อมูล (ถ้ามีกีฬาเดิม ให้ตัดออก)
        processed_df = df.copy()
        if mode == "Experienced":
            processed_df = processed_df[processed_df['Sport'] != selected_old]
            title = f"✅ กีฬาอันดับถัดไป (แนะนำต่อจาก {selected_old})"
        else:
            title = "✅ กีฬาที่เหมาะสมสำหรับคุณ"
            
        recs = processed_df.sort_values(by='Score', ascending=False).head(3)
        st.header(title)
        
        for i, (idx, row) in enumerate(recs.iterrows(), 1):
            with st.container():
                c_rank, c_info = st.columns([1, 6])
                with c_rank:
                    st.markdown(f"<h1 style='color:#4F8BF9;'>#{i}</h1>", unsafe_allow_html=True)
                with c_info:
                    st.markdown(f"### **{row['Sport']}** (Matching: {round(row['Score']*100, 1)}%)")
                    
                    # แสดงเหตุผล
                    reason = get_expert_reason(row, old_sport_row, user_req, mode)
                    st.info(f"**AI วิเคราะห์ว่า:** {reason}")
                    
                    # กราฟคุณสมบัติ
                    chart_data = pd.DataFrame({
                        'หัวข้อ': ['Cardio', 'Social', 'Budget', 'Flex', 'Str'],
                        'คะแนน': [row['Intensity'], row['Social'], row['Budget'], row['Flexibility'], row['Strength']]
                    }).set_index('หัวข้อ')
                    st.bar_chart(chart_data.T, height=150)
                st.markdown("---")
    else:
        st.info("กรุณาเลือกสถานะ ปรับ Slider แล้วกดปุ่มวิเคราะห์เพื่อดูผลลัพธ์ครับพี่สัส!")

st.divider()
st.subheader("📊 ฐานข้อมูลกีฬาที่ใช้เปรียบเทียบ (Real-time Cloud)")
st.dataframe(df.drop(columns=['Score'], errors='ignore'), use_container_width=True)

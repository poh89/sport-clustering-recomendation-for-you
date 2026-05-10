import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- [1. CONFIGURATION] ---
st.set_page_config(page_title="SportMatch AI ", layout="wide", page_icon="🏆")

# ลิงก์เชื่อมต่อ Google Sheet (CSV)
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT-CWhytgN4VrGjp82zbtzbjTrw0HtIqcUtOUuyt8HVwiJbwB9_gRBPt5MRlGUe6AGILR9W2-y8_Inf/pub?output=csv"
features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']

# ข้อมูลสำรองภายในระบบ
STATIC_DATA = {
    'Sport': ['วิ่ง', 'ว่ายน้ำ', 'โยคะ', 'ฟุตบอล', 'แบดมินตัน', 'มวยสากล', 'จักรยาน', 'ยกน้ำหนัก', 'เทนนิส', 'ปีนหน้าผา', 'ยิงธนู'],
    'Intensity': [8, 7, 3, 9, 7, 10, 6, 8, 8, 7, 2],
    'Social': [1, 1, 2, 10, 8, 2, 4, 1, 6, 5, 2],
    'Budget': [2, 4, 3, 3, 5, 4, 8, 5, 9, 7, 8],
    'Flexibility': [3, 8, 10, 5, 7, 6, 3, 2, 6, 8, 4],
    'Strength': [4, 6, 5, 6, 4, 8, 5, 10, 6, 9, 6]
}

# --- [2. FUNCTIONS: LOGIC & LOADING] ---
@st.cache_data(ttl=1)
def load_data(use_google=True):
    if not use_google:
        return pd.DataFrame(STATIC_DATA)
    try:
        df = pd.read_csv(SHEET_URL).iloc[:, 1:7].dropna(how='all')
        df.columns = ['Sport', 'Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(5)
        return df
    except:
        return pd.DataFrame(STATIC_DATA)

def get_expert_reason(new_row, old_row, user_req, is_newbie):
    reasons = []
    # วิเคราะห์ตามค่า Slider
    if user_req[0] >= 7: reasons.append(f"เน้นการฝึกหัวใจ (Cardio) ระดับ {user_req[0]} ตามที่คุณต้องการ")
    if user_req[2] <= 4: reasons.append(f"ใช้งบประมาณน้อย (ระดับ {new_row['Budget']}) เริ่มได้ทันที")
    if user_req[4] >= 7: reasons.append("ช่วยเสริมพละกำลังกล้ามเนื้อหลัก")

    # วิเคราะห์เชิงเปรียบเทียบ (ถ้ามีกีฬาเดิม)
    if not is_newbie and old_row is not None:
        if new_row['Intensity'] > old_row['Intensity']:
            reasons.append(f"ท้าทายความอึดได้มากกว่า {old_row['Sport']} เดิม")
        if new_row['Social'] > old_row['Social']:
            reasons.append(f"เพิ่มโอกาสสังคมได้ดีกว่ากีฬาเดิม")
        reasons.append(f"ช่วยอุดช่องว่างทักษะที่ {old_row['Sport']} ยังเข้าไม่ถึง")
    elif is_newbie:
        reasons.append("เป็นกีฬาที่เปิดรับมือใหม่และสร้างพื้นฐานร่างกายที่สมบูรณ์")

    return " อีกทั้งยัง ".join(reasons[:3])

# --- [3. UI INTERFACE: รูปแบบเดิม + ปุ่มเลือกแหล่งข้อมูล] ---
st.title("🏆 SportMatch AI Expert")
st.caption("ระบบวิเคราะห์กีฬาอัจฉริยะด้วย Machine Learning")

# --- ส่วนเลือกแหล่งข้อมูล (เหมือนของเดิม) ---
with st.sidebar:
    st.header("⚙️ ตั้งค่าแหล่งข้อมูล")
    source = st.radio("เลือกฐานข้อมูลที่จะใช้วิเคราะห์:", ["Google Form (Real-time)", "ข้อมูลระบบ (Static)"])
    df = load_data(use_google=(source == "Google Form (Real-time)"))
    st.divider()
    st.info("💡 ระบบจะดึงข้อมูลจากแหล่งที่เลือกมาคำนวณความคล้ายคลึง (Cosine Similarity)")

# ส่วนนำเข้าข้อมูลผู้ใช้
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("👤 ข้อมูลผู้ใช้งาน")
    experience = st.radio("พื้นฐานกีฬา:", ["ไม่มีพื้นฐาน", "เคยเล่นกีฬาบางชนิดมาก่อน"])
    
    old_sport_row = None
    bonus_vector = np.zeros(5)
    
    if experience == "เคยเล่นกีฬาบางชนิดมาก่อน":
        selected_old = st.selectbox("เลือกกีฬาที่เล่นอยู่ในปัจจุบัน:", df['Sport'].unique())
        old_sport_row = df[df['Sport'] == selected_old].iloc[0]
        bonus_vector = old_sport_row[features].values
        st.success(f"อ้างอิงทักษะจาก: {selected_old}")

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

# ส่วนแสดงผลลัพธ์
if run_btn:
    user_req = np.array([u_int, u_soc, u_bud, u_flex, u_str])
    is_newbie = (experience == "ไม่มีพื้นฐาน")
    
    # คำนวณ Vector (ทักษะเดิม 30% + ความต้องการใหม่ 70%)
    final_vector = (user_req * 0.7) + (bonus_vector * 0.3) if not is_newbie else user_req
    
    sim = cosine_similarity([final_vector], df[features])
    df['Score'] = sim[0]
    
    processed_df = df.copy()
    if not is_newbie:
        processed_df = processed_df[processed_df['Sport'] != selected_old] # ตัดกีฬาเดิม
    
    recs = processed_df.sort_values(by='Score', ascending=False).head(3)
    
    st.divider()
    st.header("🔥 รายการที่แนะนำจากฐานข้อมูล")
    
    for i, (idx, row) in enumerate(recs.iterrows(), 1):
        with st.container():
            c_rank, c_info = st.columns([1, 6])
            with c_rank:
                st.markdown(f"<h1 style='text-align:center; color:#4F8BF9; font-size:60px;'>#{i}</h1>", unsafe_allow_html=True)
            with c_info:
                st.markdown(f"### **{row['Sport']}** (Match: {round(row['Score']*100, 1)}%)")
                
                reason = get_expert_reason(row, old_sport_row, user_req, is_newbie)
                st.info(f"**AI วิเคราะห์เหตุผล:** {reason}")
                
                # กราฟคุณสมบัติ
                chart_data = pd.DataFrame({
                    'หัวข้อ': ['Cardio', 'Social', 'Budget', 'Flex', 'Str'],
                    'คะแนน': [row['Intensity'], row['Social'], row['Budget'], row['Flexibility'], row['Strength']]
                }).set_index('หัวข้อ')
                st.bar_chart(chart_data.T, height=150)
            st.markdown("---")

# ส่วนฐานข้อมูล (ย้ายมาไว้ข้างล่างเหมือนเดิม)
st.divider()
st.subheader(f"📊 ตรวจสอบฐานข้อมูล ({source})")
st.dataframe(df.drop(columns=['Score'], errors='ignore'), use_container_width=True)

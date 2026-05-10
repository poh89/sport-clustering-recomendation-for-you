import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- [1. CONFIGURATION] ---
st.set_page_config(page_title="SportMatch AI Expert", layout="wide", page_icon="🏆")

# ลิงก์เชื่อมต่อ Google Sheet (CSV)
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT-CWhytgN4VrGjp82zbtzbjTrw0HtIqcUtOUuyt8HVwiJbwB9_gRBPt5MRlGUe6AGILR9W2-y8_Inf/pub?output=csv"
features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']

# --- [2. FUNCTIONS: LOGIC & REASONING] ---
@st.cache_data(ttl=1)
def load_data():
    try:
        # ดึงข้อมูลจาก Google Sheet และจัดการค่า Error อัตโนมัติ
        df = pd.read_csv(SHEET_URL).iloc[:, 1:7].dropna(how='all')
        df.columns = ['Sport', 'Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(5)
        return df
    except:
        # ข้อมูลสำรองกรณีลิงก์เสีย
        return pd.DataFrame({
            'Sport': ['วิ่ง', 'ว่ายน้ำ', 'โยคะ', 'ฟุตบอล', 'แบดมินตัน'],
            'Intensity': [8, 7, 3, 9, 7], 'Social': [1, 1, 2, 10, 8],
            'Budget': [2, 4, 3, 3, 5], 'Flexibility': [3, 8, 10, 5, 7], 'Strength': [4, 6, 5, 6, 4]
        })

def get_detailed_analysis(new_row, old_row, user_req, is_newbie):
    reasons = []
    
    # ส่วนที่ 1: วิเคราะห์ตาม Slider (ความต้องการปัจจุบัน)
    if user_req[0] >= 7: reasons.append(f"เน้นการฝึกหัวใจ (Cardio) ระดับ {user_req[0]} ตามเป้าหมายของคุณ")
    if user_req[2] <= 4: reasons.append(f"ใช้งบประมาณต่ำ (ระดับ {new_row['Budget']}) เริ่มต้นได้ง่ายไม่กดดัน")
    if user_req[4] >= 7: reasons.append("ช่วยเสริมสร้างพละกำลังและมวลกล้ามเนื้อหลัก")

    # ส่วนที่ 2: วิเคราะห์เชิงเปรียบเทียบ (ถ้าเคยเล่นกีฬาเดิม)
    if not is_newbie and old_row is not None:
        if new_row['Intensity'] > old_row['Intensity']:
            reasons.append(f"ท้าทายความอึดได้มากกว่า {old_row['Sport']} ที่คุณเล่นอยู่")
        if new_row['Social'] > old_row['Social']:
            reasons.append(f"เพิ่มโอกาสเข้าสังคมและทำงานเป็นทีมได้ดีกว่าเดิม")
        reasons.append(f"ช่วยอุดช่องว่างและพัฒนาทักษะส่วนที่ {old_row['Sport']} ยังเข้าไม่ถึง")
    
    # ส่วนที่ 3: กรณีมือใหม่
    elif is_newbie:
        reasons.append(f"เป็นกีฬาที่เปิดรับมือใหม่และสร้างพื้นฐานร่างกายที่สมดุล")

    return " อีกทั้งยัง ".join(reasons[:3])

# --- [3. UI INTERFACE: รูปแบบเดิมที่พี่ต้องการ] ---
st.title("🏆 SportMatch AI Expert")
st.caption("ระบบวิเคราะห์กีฬาที่แม่นยำที่สุดด้วยฐานข้อมูลและการคำนวณเชิงคณิตศาสตร์")

df = load_data()

# แถวบน: ส่วนนำเข้าข้อมูล
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("👤 ข้อมูลผู้ใช้งาน")
    experience = st.radio("สถานะพื้นฐาน:", ["ไม่มีพื้นฐาน", "เคยเล่นกีฬาบางชนิดมาก่อน"])
    
    old_sport_row = None
    bonus_vector = np.zeros(5)
    
    if experience == "เคยเล่นกีฬาบางชนิดมาก่อน":
        selected_old = st.selectbox("เลือกกีฬาที่เล่นอยู่ในปัจจุบัน:", df['Sport'].unique())
        old_sport_row = df[df['Sport'] == selected_old].iloc[0]
        bonus_vector = old_sport_row[features].values
        st.success(f"อ้างอิงฐานข้อมูลจาก: {selected_old}")

with col_right:
    st.subheader("🎯 เป้าหมายตาม Slider")
    c1, c2 = st.columns(2)
    with c1:
        u_int = st.select_slider("1. ความเหนื่อย (Cardio)", options=range(1, 11), value=5)
        u_soc = st.select_slider("2. การเข้าสังคม (Social)", options=range(1, 11), value=5)
        u_bud = st.select_slider("3. งบประมาณ (1=ประหยัด)", options=range(1, 11), value=5)
    with c2:
        u_flex = st.select_slider("4. ความยืดหยุ่น", options=range(1, 11), value=5)
        u_str = st.select_slider("5. พละกำลัง", options=range(1, 11), value=5)
    
    run_btn = st.button("🚀 เริ่มการวิเคราะห์กีฬา", use_container_width=True)

# แถวกลาง: ผลการวิเคราะห์
if run_btn:
    user_req = np.array([u_int, u_soc, u_bud, u_flex, u_str])
    is_newbie = (experience == "ไม่มีพื้นฐาน")
    
    # คำนวณความคล้ายคลึง (Hybrid: ทักษะเดิม 30% + ความต้องการใหม่ 70%)
    final_vector = (user_req * 0.7) + (bonus_vector * 0.3) if not is_newbie else user_req
    
    sim = cosine_similarity([final_vector], df[features])
    df['Score'] = sim[0]
    
    # กรองข้อมูล
    processed_df = df.copy()
    if not is_newbie:
        processed_df = processed_df[processed_df['Sport'] != selected_old] # ตัดกีฬาเดิมทิ้ง
    
    recs = processed_df.sort_values(by='Score', ascending=False).head(3)
    
    st.divider()
    st.header("🔥 รายการที่แนะนำจากฐานข้อมูล")
    
    for i, (idx, row) in enumerate(recs.iterrows(), 1):
        with st.container():
            c_rank, c_info = st.columns([1, 6])
            with c_rank:
                st.markdown(f"<h1 style='text-align:center; color:#4F8BF9; font-size:60px;'>#{i}</h1>", unsafe_allow_html=True)
            with c_info:
                st.markdown(f"### **{row['Sport']}** (ความแม่นยำ {round(row['Score']*100, 1)}%)")
                
                # แสดงเหตุผลแบบยาวที่ดึงข้อมูลมาวิเคราะห์
                reason = get_detailed_analysis(row, old_sport_row, user_req, is_newbie)
                st.info(f"**วิเคราะห์จากเป้าหมายของคุณ:** {reason}")
                
                # กราฟคุณสมบัติ
                chart_data = pd.DataFrame({
                    'หัวข้อ': ['Cardio', 'Social', 'Budget', 'Flex', 'Str'],
                    'คะแนน': [row['Intensity'], row['Social'], row['Budget'], row['Flexibility'], row['Strength']]
                }).set_index('หัวข้อ')
                st.bar_chart(chart_data.T, height=150)
            st.markdown("---")

# แถวล่างสุด: ฐานข้อมูล
st.divider()
st.subheader("📊 ฐานข้อมูลที่ใช้วิเคราะห์ (จาก Google Sheet)")
st.dataframe(df.drop(columns=['Score'], errors='ignore'), use_container_width=True)

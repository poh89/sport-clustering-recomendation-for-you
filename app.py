import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- [1. CONFIGURATION] ---
st.set_page_config(page_title="SportMatch AI Expert", layout="wide", page_icon="🏆")
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT-CWhytgN4VrGjp82zbtzbjTrw0HtIqcUtOUuyt8HVwiJbwB9_gRBPt5MRlGUe6AGILR9W2-y8_Inf/pub?output=csv"
features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']

# --- [2. FUNCTIONS] ---
@st.cache_data(ttl=1)
def load_data(use_google=True):
    try:
        df = pd.read_csv(SHEET_URL).iloc[:, 1:7].dropna(how='all')
        df.columns = ['Sport', 'Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(5)
        return df
    except:
        return pd.DataFrame({'Sport':['วิ่ง','เทนนิส','โยคะ'], 'Intensity':[9,7,3], 'Social':[1,6,2], 'Budget':[2,8,3], 'Flexibility':[3,6,10], 'Strength':[5,6,5]})

def get_deep_reason(row, old_row, user_req, is_newbie):
    reasons = []
    if user_req[0] >= 7:
        reasons.append(f"เน้นการฝึกหัวใจ (Cardio) ระดับ {user_req[0]} ตามเป้าหมายของคุณ")
    elif user_req[0] <= 3:
        reasons.append(f"ตอบโจทย์ความเหนื่อยระดับต่ำ ช่วยให้เล่นได้ต่อเนื่องโดยไม่ล้าสะสม")
    if user_req[2] <= 4:
        reasons.append(f"ใช้งบประมาณต่ำ (ระดับ {row['Budget']}) เริ่มได้ทันที")

    if not is_newbie and old_row is not None:
        reasons.append(f"เป็นการนำทักษะจาก {old_row['Sport']} มาต่อยอดเพื่ออุดช่องว่างการพัฒนาที่กีฬาเดิมยังเข้าไม่ถึง")
    elif is_newbie:
        reasons.append(f"สำหรับมือใหม่ {row['Sport']} คือจุดเริ่มต้นที่สมดุลที่สุดในการสร้างพื้นฐานร่างกาย")
    return " อีกทั้งยัง ".join(reasons)

# --- [3. UI INTERFACE] ---
with st.sidebar:
    st.header("⚙️ ตั้งค่าแหล่งข้อมูล")
    source = st.radio("เลือกแหล่งข้อมูลที่จะใช้:", ["Google Form (Real-time)", "ข้อมูลระบบ (Static)"])
    df = load_data(use_google=(source == "Google Form (Real-time)"))

st.title("🏆 SportMatch AI Expert")
st.caption("ระบบวิเคราะห์กีฬาอัจฉริยะ (Professional Adaptive Reasoning)")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("👤 ข้อมูลผู้ใช้งาน")
    experience = st.radio("พื้นฐานกีฬา:", ["ไม่มีพื้นฐาน", "เคยเล่นกีฬาบางชนิดมาก่อน"])
    old_sport_row = None
    bonus_vector = np.zeros(5)
    if experience == "เคยเล่นกีฬาบางชนิดมาก่อน":
        selected_old = st.selectbox("เลือกกีฬาปัจจุบัน:", df['Sport'].unique())
        old_sport_row = df[df['Sport'] == selected_old].iloc[0]
        bonus_vector = old_sport_row[features].values

with col_right:
    st.subheader("🎯 เป้าหมายที่คุณต้องการ")
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
    st.header("🔥 ผลการวิเคราะห์และกีฬาที่แนะนำ")
    
    for i, (idx, row) in enumerate(recs.iterrows(), 1):
        with st.container():
            c_rank, c_info = st.columns([0.8, 5.2])
            with c_rank:
                st.markdown(f"<h1 style='color:#1E3A8A; font-size:60px;'>#{i}</h1>", unsafe_allow_html=True)
            with c_info:
                st.markdown(f"### **{row['Sport']}** (Match Score: {round(row['Score']*100, 1)}%)")
                
                reason = get_deep_reason(row, old_sport_row, user_req, is_newbie)
                st.markdown(f"""
                <div style="background-color: #F0F4F8; padding: 15px; border-radius: 10px; border-left: 6px solid #1E3A8A; font-size: 16px;">
                    <strong>🧠 บทวิเคราะห์โดย AI:</strong> {reason}
                </div>
                """, unsafe_allow_html=True)
                
                # --- [แก้ไขกราฟ: แนวนอน แท่งเล็ก สีสวย] ---
                chart_df = pd.DataFrame({
                    'คุณสมบัติ': ['งบประมาณ', 'เหนื่อย(Cardio)', 'ยืดหยุ่น', 'สังคม', 'พละกำลัง'],
                    'คะแนน': [row['Budget'], row['Intensity'], row['Flexibility'], row['Social'], row['Strength']]
                })
                
                # ใช้ Horizontal Bar Chart เพื่อให้ตัวหนังสือเป็นแนวนอนและอ่านง่าย
                st.bar_chart(data=chart_df.set_index('คุณสมบัติ'), height=250, use_container_width=True)
                
            st.markdown("---")

st.divider()
st.subheader("📊 ตารางข้อมูลอ้างอิง")
st.dataframe(df.drop(columns=['Score'], errors='ignore'), use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt

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

# --- [🔥 รื้อระบบวิเคราะห์ใหม่หมด ฉลาดและสอดคล้อง Slider 100%] ---
def get_deep_reason(row, old_row, user_req, is_newbie):
    reason_text = ""

    # 1. ตอบคำถาม: "เหตุผลเริ่มต้น ทำไมต้องเริ่ม / ทำไมต้องเปลี่ยน"
    if is_newbie:
        reason_text += f"🔰 <b>ทำไมมือใหม่ถึงควรเริ่ม:</b> {row['Sport']} เป็นกีฬาที่ปรับระดับความเข้มข้นได้ตามร่างกายคุณ ช่วยลดความกังวลเรื่องอาการบาดเจ็บ และเป็นจุดเริ่มต้นที่ดีที่สุดในการปูพื้นฐานความฟิต <br><br>"
    else:
        reason_text += f"🔄 <b>ทำไมถึงควรเปลี่ยนมาท้าทายด้วย {row['Sport']}:</b> เพื่อทะลวงกำแพงความจำเจจาก {old_row['Sport']} กีฬาชนิดนี้จะบังคับให้คุณใช้สรีระและกล้ามเนื้อในมิติใหม่ๆ ที่กีฬาเดิมยังให้ไม่ได้ <br><br>"

    # 2. ตอบคำถาม: "สอดคล้องกับ Slider ยังไง และ ส่งเสริมอะไร"
    reason_text += "🎯 <b>สอดคล้องกับเป้าหมายของคุณ:</b> "
    benefits = []

    # วิเคราะห์ Cardio
    if user_req[0] >= 6:
        benefits.append(f"ตามที่คุณตั้ง Slider <b>ความเหนื่อยไว้สูง ({user_req[0]}/10)</b> การเล่นกีฬานี้จะเข้าไป <u>ส่งเสริมระบบไหลเวียนโลหิตและเพิ่มความจุของปอด (Cardio)</u> ให้แข็งแรงทะลุขีดจำกัด")
    elif user_req[0] <= 4:
        benefits.append(f"ด้วย <b>ความเหนื่อยที่คุณเลือกไว้ต่ำ ({user_req[0]}/10)</b> กีฬานี้ตอบโจทย์มาก เพราะ <u>ส่งเสริมให้ร่างกายได้ฟื้นฟู (Active Recovery)</u> โดยไม่สร้างความตึงเครียดให้หัวใจมากเกินไป")

    # วิเคราะห์ Social
    if user_req[1] >= 6:
        benefits.append(f"จากความต้องการ <b>เข้าสังคมระดับ {user_req[1]}</b> กีฬานี้จะ <u>ส่งเสริมทักษะการสื่อสาร การทำงานเป็นทีม</u> และขยายคอนเนคชันใหม่ๆ ให้คุณ")
    elif user_req[1] <= 4:
        benefits.append(f"คุณต้องการ <b>ความเป็นส่วนตัว (สังคมระดับ {user_req[1]})</b> กีฬานี้จะ <u>ส่งเสริมสมาธิและการโฟกัส</u> ให้คุณได้อยู่กับตัวเองเพื่อทบทวนความคิดได้อย่างเต็มที่")

    # วิเคราะห์ Budget
    if user_req[2] <= 4:
        benefits.append(f"สอดคล้องกับ <b>งบประมาณจำกัด ({user_req[2]}/10)</b> กีฬานี้ <u>ส่งเสริมวินัยทางการเงิน</u> เพราะใช้อุปกรณ์น้อย เริ่มได้ทันทีแบบไม่เจ็บกระเป๋า")

    # วิเคราะห์ Flex / Strength
    if user_req[4] >= 6 or row['Strength'] >= 7:
        benefits.append(f"นอกจากนี้ยัง <u>ส่งเสริมความแข็งแกร่งของมวลกล้ามเนื้อหลัก (Core Strength)</u> ทำให้โครงสร้างร่างกายคุณมั่นคงขึ้น")
    elif user_req[3] >= 6 or row['Flexibility'] >= 7:
        benefits.append(f"และช่วย <u>ส่งเสริมความยืดหยุ่นของเส้นเอ็น</u> ลดความเสี่ยงออฟฟิศซินโดรมได้อย่างชะงัด")

    # กรณีปรับ Slider ไว้ตรงกลางหมด (5)
    if not benefits:
        benefits.append(f"คุณสมบัติของกีฬานี้มีความสมดุลในทุกมิติตามที่คุณตั้งค่าไว้ ซึ่งจะ <u>ส่งเสริมสุขภาพโดยรวม</u> ทั้งความอึดและความยืดหยุ่นไปพร้อมๆ กัน")

    # รวมประโยคเข้าด้วยกัน
    reason_text += " และ ".join(benefits[:3])

    return reason_text

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
    
    for i, (idx, row) in enumerate(recs.iterrows(), 1):
        with st.container():
            st.markdown(f"### **{row['Sport']} (Match Score: {round(row['Score']*100, 1)}%)**")
            
            reason = get_deep_reason(row, old_sport_row, user_req, is_newbie)
            
            # กรอบวิเคราะห์แบบจัดเต็ม แถบสีน้ำเงินดูแพง
            st.markdown(f"""
            <div style="background-color: #F8FAFC; padding: 20px; border-radius: 10px; border-left: 6px solid #1E3A8A; font-size: 15px; margin-bottom: 20px; line-height: 1.6; color: #334155;">
                {reason}
            </div>
            """, unsafe_allow_html=True)
            
            # --- [กราฟแท่งแบบ แนวนอน 100% + แท่งเล็ก + สีเดิมที่พี่ชอบ] ---
            chart_df = pd.DataFrame({
                'Attributes': ['Budget', 'Cardio', 'Flexibility', 'Social', 'Strength'],
                'Score': [row['Budget'], row['Intensity'], row['Flexibility'], row['Social'], row['Strength']]
            })
            
            chart = alt.Chart(chart_df).mark_bar(
                size=18,           
                color='#1E3A8A'     
            ).encode(
                x=alt.X('Attributes:N', title=None, axis=alt.Axis(labelAngle=0)), 
                y=alt.Y('Score:Q', title='คะแนน', scale=alt.Scale(domain=[0, 10]))
            ).properties(height=280)
            
            st.altair_chart(chart, use_container_width=True)
            st.markdown("---")

st.divider()
st.subheader("📊 ตารางข้อมูลอ้างอิง")
st.dataframe(df.drop(columns=['Score'], errors='ignore'), use_container_width=True)

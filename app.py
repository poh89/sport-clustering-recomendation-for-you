import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt

# --- [1. CONFIGURATION] ---
st.set_page_config(page_title="SportMatch AI Expert", layout="wide", page_icon="🏆")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT-CWhytgN4VrGjp82zbtzbjTrw0HtIqcUtOUuyt8HVwiJbwB9_gRBPt5MRlGUe6AGILR9W2-y8_Inf/pub?output=csv"

features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']

# --- [2. LOAD DATA] ---
@st.cache_data(ttl=1)
def load_data():
    try:
        raw_df = pd.read_csv(SHEET_URL)

        # จับคอลัมน์ A ถึง I (index 0-8) = 9 คอลัมน์
        df = raw_df.iloc[:, 0:9].copy()
        df.columns = ['Sport', 'Intensity', 'Social', 'Budget', 'Flexibility', 'Strength',
                      'Complexity', 'Physiological', 'Risk']

        # ลบแถวที่ชื่อกีฬาว่าง
        df['Sport'] = df['Sport'].astype(str)
        df = df[df['Sport'].str.strip() != '']
        df = df[df['Sport'].str.lower() != 'nan']
        df = df.reset_index(drop=True)

        # แปลงค่าตัวเลข
        numeric_cols = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength',
                        'Complexity', 'Physiological', 'Risk']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # ตรวจจับค่าว่าง TDS
        tds_cols = ['Complexity', 'Physiological', 'Risk']
        missing_tds = df[df[tds_cols].isnull().any(axis=1)]
        if len(missing_tds) > 0:
            st.warning(f"⚠️ พบ {len(missing_tds)} กีฬาที่ยังไม่มีค่า Complexity/Physiological/Risk: {missing_tds['Sport'].tolist()}")

        # fillna
        for col in ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']:
            df[col] = df[col].fillna(5)
        for col in tds_cols:
            df[col] = df[col].fillna(8)

        return df

    except Exception as e:
        st.error(f"❌ โหลดข้อมูลไม่สำเร็จ: {e}")
        return pd.DataFrame({
            'Sport': ['วิ่ง', 'เทนนิส', 'โยคะ'],
            'Intensity': [9, 7, 3], 'Social': [1, 6, 2],
            'Budget': [2, 8, 3], 'Flexibility': [3, 6, 10],
            'Strength': [5, 6, 5], 'Complexity': [3, 6, 4],
            'Physiological': [8, 6, 3], 'Risk': [2, 4, 1]
        })


# --- [3. DEEP REASON] ---
def get_deep_reason(row, old_row, user_req, is_newbie, gender, age):
    reason_text = ""

    if is_newbie:
        reason_text += f"🔰 <b>ทำไมมือใหม่ (เพศ{gender} อายุ {age} ปี) ถึงควรเริ่ม:</b> {row['Sport']} เป็นกีฬาที่ปรับระดับความเข้มข้นได้ตามร่างกายคุณ ช่วยลดความกังวลเรื่องอาการบาดเจ็บ และเป็นจุดเริ่มต้นที่ดีที่สุดในการปูพื้นฐานความฟิต <br><br>"
    else:
        reason_text += f"🔄 <b>ทำไมถึงควรเปลี่ยนมาท้าทายด้วย {row['Sport']}:</b> เพื่อทะลวงกำแพงความจำเจจาก {old_row['Sport']} สำหรับผู้ใช้เพศ{gender} วัย {age} ปี กีฬาชนิดนี้จะบังคับให้คุณใช้สรีระและกล้ามเนื้อในมิติใหม่ๆ ที่กีฬาเดิมยังให้ไม่ได้ <br><br>"

    if age >= 40:
        reason_text += f"🦴 <b>ข้อแนะนำด้านสรีรวิทยา (วัย {age} ปี):</b> กีฬา {row['Sport']} ช่วยรักษามวลกล้ามเนื้อและเพิ่มความยืดหยุ่น ซึ่งจำเป็นอย่างยิ่งสำหรับการป้องกันภาวะข้อต่อเสื่อมในวัยของคุณ <br><br>"
    elif age <= 22:
        reason_text += f"⚡ <b>การส่งเสริมศักยภาพร่างกาย (วัยรุ่น/วัยเรียน):</b> กีฬานี้จะช่วยกระตุ้นการเผาผลาญและสร้างมวลกล้ามเนื้อหลัก (Core Muscle) ได้อย่างรวดเร็วในช่วงที่ร่างกายกำลังพัฒนา <br><br>"

    reason_text += "🎯 <b>สอดคล้องกับเป้าหมายของคุณ:</b> "
    benefits = []

    if user_req[0] >= 6:
        benefits.append(f"ตามที่คุณตั้ง Slider <b>ความเหนื่อยไว้สูง ({user_req[0]}/10)</b> การเล่นกีฬานี้จะเข้าไป <u>ส่งเสริมระบบไหลเวียนโลหิตและเพิ่มความจุของปอด (Cardio)</u> ให้แข็งแรงทะลุขีดจำกัด")
    elif user_req[0] <= 4:
        benefits.append(f"ด้วย <b>ความเหนื่อยที่คุณเลือกไว้ต่ำ ({user_req[0]}/10)</b> กีฬานี้ตอบโจทย์มาก เพราะ <u>ส่งเสริมให้ร่างกายได้ฟื้นฟู (Active Recovery)</u> โดยไม่สร้างความตึงเครียดให้หัวใจมากเกินไป")

    if user_req[1] >= 6:
        benefits.append(f"จากความต้องการ <b>เข้าสังคมระดับ {user_req[1]}</b> กีฬานี้จะ <u>ส่งเสริมทักษะการสื่อสาร การทำงานเป็นทีม</u> และขยายคอนเนคชันใหม่ๆ ให้คุณ")
    elif user_req[1] <= 4:
        benefits.append(f"คุณต้องการ <b>ความเป็นส่วนตัว (สังคมระดับ {user_req[1]})</b> กีฬานี้จะ <u>ส่งเสริมสมาธิและการโฟกัส</u> ให้คุณได้อยู่กับตัวเองเพื่อทบทวนความคิดได้อย่างเต็มที่")

    if user_req[2] <= 4:
        benefits.append(f"สอดคล้องกับ <b>งบประมาณจำกัด ({user_req[2]}/10)</b> กีฬานี้ <u>ส่งเสริมวินัยทางการเงิน</u> เพราะใช้อุปกรณ์น้อย เริ่มได้ทันทีแบบไม่เจ็บกระเป๋า")

    if user_req[4] >= 6 or row['Strength'] >= 7:
        benefits.append(f"นอกจากนี้ยัง <u>ส่งเสริมความแข็งแกร่งของมวลกล้ามเนื้อหลัก (Core Strength)</u> ทำให้โครงสร้างร่างกายคุณมั่นคงขึ้น")
    elif user_req[3] >= 6 or row['Flexibility'] >= 7:
        benefits.append(f"และช่วย <u>ส่งเสริมความยืดหยุ่นของเส้นเอ็น</u> ลดความเสี่ยงออฟฟิศซินโดรมได้อย่างชะงัด")

    if not benefits:
        benefits.append(f"คุณสมบัติของกีฬานี้มีความสมดุลในทุกมิติตามที่คุณตั้งค่าไว้ ซึ่งจะ <u>ส่งเสริมสุขภาพโดยรวม</u> ทั้งความอึดและความยืดหยุ่นไปพร้อมๆ กัน")

    reason_text += " และ ".join(benefits[:3])
    return reason_text


# --- [4. UI] ---
with st.sidebar:
    st.header("⚙️ ตั้งค่าระบบ")
    st.info("📡 เชื่อมต่อ Google Sheets แบบ Real-time")
    if st.button("🗑️ ล้าง Cache และรีเฟรชข้อมูล"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption("💡 กดล้าง Cache หลังแก้ไขข้อมูลใน Google Sheets")

df = load_data()

st.title("🏆 SportMatch AI Expert")
st.caption("ระบบวิเคราะห์กีฬาอัจฉริยะ (Professional Adaptive Reasoning)")
st.success(f"✅ โหลดข้อมูลสำเร็จ: {len(df)} กีฬา")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("👤 ข้อมูลผู้ใช้งาน")
    gender = st.selectbox("ระบุเพศของคุณ:", ["ชาย", "หญิง", "ไม่ระบุ"])
    age = st.number_input("ระบุอายุของคุณ (ปี):", min_value=1, max_value=100, value=20, step=1)
    st.divider()
    experience = st.radio("พื้นฐานกีฬา:", ["ไม่มีพื้นฐาน (มือใหม่)", "เคยเล่นกีฬามาก่อน (ต้องการเปลี่ยน)"])

    old_sport_row = None
    bonus_vector = np.zeros(5)

    if experience == "เคยเล่นกีฬามาก่อน (ต้องการเปลี่ยน)":
        selected_old = st.selectbox("เลือกกีฬาที่เล่นอยู่ปัจจุบัน:", df['Sport'].unique())
        old_sport_row = df[df['Sport'] == selected_old].iloc[0]
        bonus_vector = old_sport_row[features].values

with col_right:
    st.subheader("🎯 เป้าหมายที่คุณต้องการ")
    c1, c2 = st.columns(2)
    with c1:
        u_int  = st.select_slider("1. ความเหนื่อย (Cardio)",  options=range(1, 11), value=5)
        u_soc  = st.select_slider("2. การเข้าสังคม (Social)", options=range(1, 11), value=5)
        u_bud  = st.select_slider("3. งบประมาณ (Budget)",     options=range(1, 11), value=5)
    with c2:
        u_flex = st.select_slider("4. ความยืดหยุ่น",          options=range(1, 11), value=5)
        u_str  = st.select_slider("5. พละกำลัง",              options=range(1, 11), value=5)

    run_btn = st.button("🚀 เริ่มการวิเคราะห์", use_container_width=True)


# --- [5. PROCESSING] ---
if run_btn:
    user_req     = np.array([u_int, u_soc, u_bud, u_flex, u_str], dtype=float)
    is_newbie    = (experience == "ไม่มีพื้นฐาน (มือใหม่)")
    final_vector = (user_req * 0.7) + (bonus_vector * 0.3) if not is_newbie else user_req

    # คำนวณ Cosine Similarity
    sim = cosine_similarity([final_vector], df[features].values)
    df['Score'] = sim[0]

    processed_df = df.copy()

    # ตัดกีฬาเดิมออก
    if not is_newbie:
        processed_df = processed_df[processed_df['Sport'] != selected_old]

    # TDS Filter สำหรับมือใหม่
    if is_newbie:
        processed_df['TDS'] = (
            processed_df['Complexity']    * 0.4 +
            processed_df['Physiological'] * 0.3 +
            processed_df['Risk']          * 0.3
        )

        with st.expander("🔍 Debug: กีฬาที่ถูก TDS Filter ตัดออก (TDS ≥ 7)"):
            filtered_out = processed_df[processed_df['TDS'] >= 7][['Sport', 'Complexity', 'Physiological', 'Risk', 'TDS']]
            filtered_out = filtered_out.sort_values('TDS', ascending=False)
            st.dataframe(filtered_out, use_container_width=True)
            st.caption(f"🚫 ถูกตัดออก: {len(filtered_out)} รายการ")

        processed_df = processed_df[processed_df['TDS'] < 7]

        if len(processed_df) == 0:
            st.error("❌ ไม่พบกีฬาที่เหมาะสมสำหรับมือใหม่ กรุณาปรับเกณฑ์ใหม่")
            st.stop()

    # Top 3
    recs = processed_df.sort_values(by='Score', ascending=False).head(3)

    st.divider()
    st.subheader("🎯 ผลการวิเคราะห์: กีฬาที่เหมาะกับคุณ")

    for i, (idx, row) in enumerate(recs.iterrows(), 1):
        with st.container():
            c_rank, c_info = st.columns([1, 6])
            with c_rank:
                medal = ["🥇", "🥈", "🥉"][i-1]
                st.markdown(f"<h1 style='text-align:center; font-size:60px; margin-top:20px;'>{medal}</h1>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align:center; color:#1E3A8A;'>#{i}</h2>", unsafe_allow_html=True)
            with c_info:
                score_pct = round(row['Score'] * 100, 1)
                st.markdown(f"### **{row['Sport']}** (Match Score: {score_pct}%)")

                reason = get_deep_reason(row, old_sport_row, user_req, is_newbie, gender, age)
                st.markdown(f"""
                <div style="background-color: #F8FAFC; padding: 20px; border-radius: 10px;
                            border-left: 6px solid #1E3A8A; font-size: 15px;
                            margin-bottom: 20px; line-height: 1.8; color: #334155;">
                    {reason}
                </div>""", unsafe_allow_html=True)

                # Grouped Bar Chart
                chart_df = pd.DataFrame({
                    'Attributes': ['Cardio', 'Social', 'Budget', 'Flexibility', 'Strength'],
                    'กีฬาแนะนำ': [row['Intensity'], row['Social'], row['Budget'], row['Flexibility'], row['Strength']],
                    'คุณต้องการ': [u_int, u_soc, u_bud, u_flex, u_str]
                })
                chart_melted = chart_df.melt('Attributes', var_name='ประเภท', value_name='คะแนน')

                chart = alt.Chart(chart_melted).mark_bar(size=16).encode(
                    x=alt.X('Attributes:N', title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('คะแนน:Q', title='คะแนน', scale=alt.Scale(domain=[0, 10])),
                    color=alt.Color('ประเภท:N', scale=alt.Scale(
                        domain=['กีฬาแนะนำ', 'คุณต้องการ'],
                        range=['#1E3A8A', '#F59E0B']
                    )),
                    xOffset='ประเภท:N'
                ).properties(height=250)

                st.altair_chart(chart, use_container_width=True)

        st.markdown("---")

# --- [6. DATA TABLE] ---
st.divider()
st.subheader("📊 ตารางข้อมูลกีฬาทั้งหมด")
display_df = df.drop(columns=['Score', 'TDS'], errors='ignore')
st.dataframe(display_df, use_container_width=True, height=400)
st.caption(f"📋 จำนวนกีฬาในระบบ: {len(df)} รายการ")

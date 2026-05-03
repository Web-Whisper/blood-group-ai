
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Blood Report System", layout="centered")

st.title("🧠 AI Blood Report Analyzer + Blood Group Predictor")

# -----------------------------
# MODEL
# -----------------------------
df = pd.read_csv("lab_dataset.csv")

X = df.drop("blood_group", axis=1)
y = df["blood_group"]

model = RandomForestClassifier()
model.fit(X, y)

# -----------------------------
# INPUT
# -----------------------------
st.header("🔢 Enter Your Lab Values")

hb = st.number_input("Hemoglobin (g/dL)")
rbc = st.number_input("RBC Count (million/µL)")
wbc = st.number_input("WBC Count (cells/µL)")
platelets = st.number_input("Platelets (/cumm)")

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("Analyze Report"):

    st.subheader("📊 Results ki Tafseel (Interpretation)")

    # Hemoglobin
    if hb < 12:
        hb_status = "Low"
    elif hb <= 17.5:
        hb_status = "Normal"
    else:
        hb_status = "High"

    # RBC
    if rbc < 4:
        rbc_status = "Low"
    elif rbc <= 5.5:
        rbc_status = "Normal"
    else:
        rbc_status = "High"

    # WBC
    if wbc < 4500:
        wbc_status = "Low"
    elif wbc <= 11000:
        wbc_status = "Normal"
    else:
        wbc_status = "High"

    # Platelets
    if platelets < 150000:
        plt_status = "Low"
    elif platelets <= 450000:
        plt_status = "Normal"
    else:
        plt_status = "High"

    # -----------------------------
    # REPORT TEXT
    # -----------------------------
    report = f"""
**Hemoglobin (Hb) - {hb} g/dL ({hb_status})**  
Yeh khoon mein oxygen le janay wali protein hai.  
Agar yeh 12 se kam ho to anemia ho sakta hai.  
Reference: Male (13.5–17.5), Female (12.0–15.5)

---

**RBC Count - {rbc} million/µL ({rbc_status})**  
Yeh surkh khoon ke khaliye hain.  
Reference: Male (4.5–5.5), Female (4.0–5.0)

---

**WBC Count - {wbc} cells/µL ({wbc_status})**  
Yeh immune system ka hissa hain. Infection mein barh jate hain.  
Reference: 4,500–11,000

---

**Platelets - {platelets}/cumm ({plt_status})**  
Yeh clotting mein madad karte hain.  
Reference: 150,000–450,000
"""

    st.markdown(report)

    # -----------------------------
    # IMPORTANT NOTES
    # -----------------------------
    st.subheader("⚠️ Important Notes")

    st.markdown("""
- Low Platelets (<150,000): Dengue risk  
- Low Hemoglobin (<12): Iron deficiency  
- High WBC (>11,000): Infection sign  
""")

    # -----------------------------
# SMART FINAL SUMMARY
# -----------------------------

issues = []

if hb_status != "Normal":
    issues.append(f"Hemoglobin {hb_status}")

if rbc_status != "Normal":
    issues.append(f"RBC {rbc_status}")

if wbc_status != "Normal":
    issues.append(f"WBC {wbc_status}")

if plt_status != "Normal":
    issues.append(f"Platelets {plt_status}")

st.subheader("🧾 Final Assessment")

if len(issues) == 0:
    st.success("✅ Aapki tamam values normal range mein hain. Report bilkul theek hai.")

elif len(issues) == 1:
    st.warning(f"⚠️ Sirf ek issue detect hua: {issues[0]}. Behtar hai monitoring rakhein.")

elif len(issues) == 2:
    st.warning(f"⚠️ Multiple issues detect huay: {', '.join(issues)}. Doctor se consult karein.")

else:
    st.error(f"🚨 Serious condition: {', '.join(issues)} abnormal hain. Jaldi doctor se rabta karein.")

    # -----------------------------
    # GRAPH
    # -----------------------------
    st.subheader("📊 Graph")

    fig, ax = plt.subplots()
    ax.bar(["Hb","RBC","WBC","Platelets"], [hb, rbc, wbc, platelets])
    st.pyplot(fig)

    # -----------------------------
    # BLOOD GROUP
    # -----------------------------
    if hb>0 and rbc>0 and wbc>0 and platelets>0:
        pred = model.predict([[hb, rbc, wbc, platelets]])
        st.subheader("🧬 Predicted Blood Group")
        st.success(pred[0])

# -----------------------------
# CHATBOT
# -----------------------------
st.subheader("🤖 AI Health Assistant")

query = st.text_input("Ask: (e.g. Mera report theek hai?)")

if st.button("Ask"):

    if query:
        q = query.lower()

        if "theek" in q or "report" in q:
            st.write("Agar sab values normal hain to report theek hai.")

        elif "hb" in q or "hemoglobin" in q:
            st.write("Hemoglobin oxygen carry karta hai. Low ho to anemia hota hai.")

        elif "platelets" in q:
            st.write("Platelets clotting ke liye hotay hain.")

        elif "wbc" in q:
            st.write("WBC infection se fight karta hai.")

        else:
            st.write("General advice: Doctor se consult karein.")

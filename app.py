
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Blood Report System", layout="centered")

st.title("🧠 AI Blood Report Analyzer + Blood Group Predictor")

# -----------------------------
# MODEL (SAFE LOAD)
# -----------------------------
@st.cache_data
def load_model():
    df = pd.read_csv("lab_dataset.csv")

    X = df.drop("blood_group", axis=1)
    y = df["blood_group"]

    model = RandomForestClassifier()
    model.fit(X, y)

    return model

model = load_model()

# -----------------------------
# INPUT
# -----------------------------
st.header("🔢 Enter Your Lab Values")

hb = st.number_input("Hemoglobin (g/dL)", min_value=0.0)
rbc = st.number_input("RBC Count (million/µL)", min_value=0.0)
wbc = st.number_input("WBC Count (cells/µL)", min_value=0.0)
platelets = st.number_input("Platelets (/cumm)", min_value=0.0)

# -----------------------------
# STATUS FUNCTION
# -----------------------------
def get_status(value, low, high):
    if value < low:
        return "Low"
    elif value <= high:
        return "Normal"
    else:
        return "High"

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("Analyze Report"):

    # STATUS CALCULATION
    hb_status = get_status(hb, 12, 17.5)
    rbc_status = get_status(rbc, 4, 5.5)
    wbc_status = get_status(wbc, 4500, 11000)
    plt_status = get_status(platelets, 150000, 450000)

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
**WBC Count - {wbc} cells/µL ({wbc_status})**  
Yeh immune system ka hissa hain. Infection mein barh jate hain.  
Reference: 4,500–11,000

---

**Platelets - {platelets}/cumm ({plt_status})**  
Yeh clotting mein madad karte hain.  
Reference: 150,000–450,000
"""

    # -----------------------------
    # CLEAN REPORT UI
    # -----------------------------
    st.subheader("📄 Blood Test Report")

    def show_line(name, value, status):
        color = "🟢" if status == "Normal" else "🟡" if status == "Low" else "🔴"
        st.markdown(f"**{name}:** {value} → {color} {status}")

    show_line("Hemoglobin", hb, hb_status)
    show_line("RBC Count", rbc, rbc_status)
    show_line("WBC Count", wbc, wbc_status)
    show_line("Platelets", platelets, plt_status)

    # -----------------------------
    # SUMMARY LOGIC
    # -----------------------------
    issues = []

    if hb_status != "Normal":
        issues.append(f"Hemoglobin ({hb_status})")

    if rbc_status != "Normal":
        issues.append(f"RBC ({rbc_status})")

    if wbc_status != "Normal":
        issues.append(f"WBC ({wbc_status})")

    if plt_status != "Normal":
        issues.append(f"Platelets ({plt_status})")

    st.subheader("🧾 Final Assessment")

    if len(issues) == 0:
        st.success("✅ All values are within normal range. Report is healthy.")
    elif len(issues) == 1:
        st.warning(f"⚠️ Minor issue detected: {issues[0]}")
    elif len(issues) == 2:
        st.warning(f"⚠️ Multiple issues: {', '.join(issues)}")
    else:
        st.error(f"🚨 Serious concern: {', '.join(issues)}")

    # -----------------------------
    # GRAPH
    # -----------------------------
    st.subheader("📊 Visual Report")

    fig, ax = plt.subplots()
    ax.bar(["Hb","RBC","WBC","Platelets"], [hb, rbc, wbc, platelets])
    st.pyplot(fig)

    # -----------------------------
    # BLOOD GROUP PREDICTION (SAFE)
    # -----------------------------
    if hb > 0 and rbc > 0 and wbc > 0 and platelets > 0:
        pred = model.predict([[hb, rbc, wbc, platelets]])
        st.subheader("🧬 Predicted Blood Group")
        st.success(f"Your Blood Group: {pred[0]}")

# -----------------------------
# CHATBOT (IMPROVED)
# -----------------------------
st.subheader("🤖 AI Health Assistant")

query = st.text_input("Ask your question")

if st.button("Ask AI"):

    if query:
        q = query.lower()

        if "report" in q or "theek" in q:
            st.info("If all values are normal, your report is healthy.")

        elif "hemoglobin" in q or "hb" in q:
            st.info("Hemoglobin carries oxygen in blood. Low level indicates anemia.")

        elif "platelets" in q:
            st.info("Platelets help in blood clotting.")

        elif "wbc" in q:
            st.info("WBC protects body from infections.")

        else:
            st.warning("Please consult a doctor for detailed medical advice.")

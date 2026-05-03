
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.title("🧠 AI Blood Report Analyzer + Chatbot")

# Load dataset
df = pd.read_csv("lab_dataset.csv")

X = df.drop("blood_group", axis=1)
y = df["blood_group"]

model = RandomForestClassifier()
model.fit(X, y)

# -----------------------------
# INPUT SECTION
# -----------------------------
st.header("🔢 Enter Lab Values")

hb = st.number_input("Hemoglobin (g/dL)")
rbc = st.number_input("RBC Count (million/µL)")
wbc = st.number_input("WBC Count (cells/µL)")
platelets = st.number_input("Platelets (/cumm)")

# -----------------------------
# ANALYSIS
# -----------------------------
if st.button("Analyze Report"):

    st.subheader("📊 Interpretation")

    report_status = "Normal"

    # Hemoglobin
    if hb < 12:
        st.warning("Low Hemoglobin → Anemia")
        report_status = "Issue"
    elif hb <= 17.5:
        st.success("Hemoglobin Normal")
    else:
        st.error("High Hemoglobin")
        report_status = "Issue"

    # RBC
    if rbc < 4:
        st.warning("Low RBC")
        report_status = "Issue"
    elif rbc <= 5.5:
        st.success("RBC Normal")
    else:
        st.error("High RBC")
        report_status = "Issue"

    # WBC
    if wbc < 4500:
        st.warning("Low WBC")
        report_status = "Issue"
    elif wbc <= 11000:
        st.success("WBC Normal")
    else:
        st.error("High WBC → Infection")
        report_status = "Issue"

    # Platelets
    if platelets < 150000:
        st.warning("Low Platelets → Dengue risk")
        report_status = "Issue"
    elif platelets <= 450000:
        st.success("Platelets Normal")
    else:
        st.error("High Platelets")
        report_status = "Issue"

    # -----------------------------
    # AI CHATBOT RESPONSE
    # -----------------------------
    st.subheader("🤖 AI Health Assistant")

    user_query = st.text_input("Ask something (e.g. Mera report theek hai?)")

    if user_query:
        if "theek" in user_query.lower():
            if report_status == "Normal":
                st.success("✅ Aapka report bilkul normal lag raha hai.")
            else:
                st.warning("⚠️ Report mein kuch issues hain, doctor se consult karein.")

        elif "hemoglobin" in user_query.lower():
            st.info("Hemoglobin oxygen carry karta hai. Low ho to anemia hota hai.")

        elif "platelets" in user_query.lower():
            st.info("Platelets clotting ke liye zaroori hain. Dengue mein kam ho jate hain.")

        else:
            st.info("🤖 General advice: Apni report doctor ko bhi dikhaein.")

    # -----------------------------
    # GRAPHS (HEALTH VISUALIZATION)
    # -----------------------------
    st.subheader("📊 Health Graph")

    labels = ["Hb", "RBC", "WBC", "Platelets"]
    values = [hb, rbc, wbc, platelets]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Your Health Values")

    st.pyplot(fig)

    # -----------------------------
    # BLOOD GROUP PREDICTION
    # -----------------------------
    if hb > 0 and rbc > 0 and wbc > 0 and platelets > 0:
        pred = model.predict([[hb, rbc, wbc, platelets]])
        st.subheader("🧬 Predicted Blood Group")
        st.success(pred[0])

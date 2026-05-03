
import streamlit as st 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from groq import Groq

st.set_page_config(page_title="AI Blood Report System", layout="centered")

st.title("🧠 AI Blood Report Analyzer + Blood Group Predictor")

# -----------------------------
# OPENAI CLIENT (REAL AI)
# -----------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

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

    hb_status = get_status(hb, 12, 17.5)
    rbc_status = get_status(rbc, 4, 5.5)
    wbc_status = get_status(wbc, 4500, 11000)
    plt_status = get_status(platelets, 150000, 450000)

    # -----------------------------
    # REPORT UI
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
    # FINAL SUMMARY
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
        st.success("✅ All values are normal. Report is healthy.")
    elif len(issues) == 1:
        st.warning(f"⚠️ Minor issue: {issues[0]}")
    elif len(issues) == 2:
        st.warning(f"⚠️ Multiple issues: {', '.join(issues)}")
    else:
        st.error(f"🚨 Serious condition: {', '.join(issues)}")

    # -----------------------------
    # GRAPH
    # -----------------------------
    st.subheader("📊 Visual Report")

    fig, ax = plt.subplots()
    ax.bar(["Hb","RBC","WBC","Platelets"], [hb, rbc, wbc, platelets])
    st.pyplot(fig)

    # -----------------------------
    # BLOOD GROUP PREDICTION
    # -----------------------------
    if hb > 0 and rbc > 0 and wbc > 0 and platelets > 0:
        pred = model.predict([[hb, rbc, wbc, platelets]])
        st.subheader("🧬 Predicted Blood Group")
        st.success(f"Your Blood Group: {pred[0]}")

# -----------------------------
# 🤖 REAL AI CHATBOT (FIXED + SMART)
# -----------------------------
st.subheader("🤖 AI Health Assistant")

query = st.text_input("Ask anything about your report or health")

def ask_ai(question):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. Explain in simple language."
                },
                {
                    "role": "user",
                    "content": f"""
Patient Report:
Hemoglobin: {hb}
RBC: {rbc}
WBC: {wbc}
Platelets: {platelets}

Question: {question}
"""
                }
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error in AI response: {str(e)}"
        if st.button("Ask AI"):
    if query:
        answer = ask_ai(query)
        st.success(answer)

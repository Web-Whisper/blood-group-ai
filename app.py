
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("🧠 AI Blood Report Analyzer + Blood Group Predictor")

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
# FILE UPLOAD
# -----------------------------
st.header("📤 Upload Medical Report (Optional)")
uploaded_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    st.success("File uploaded successfully ✅")
    st.info("⚠️ (Advanced feature: OCR integration required for auto-reading)")

# -----------------------------
# ANALYSIS BUTTON
# -----------------------------
if st.button("Analyze Report"):

    st.subheader("📊 Results Interpretation")

    # Hemoglobin
    if hb > 0:
        if hb < 12:
            st.warning(f"Hemoglobin: {hb} (Low) → Khoon ki kami (Anemia)")
        elif hb <= 17.5:
            st.success(f"Hemoglobin: {hb} (Normal) → Oxygen carry karta hai")
        else:
            st.error(f"Hemoglobin: {hb} (High) → Dehydration ya dusra masla")

    # RBC
    if rbc > 0:
        if rbc < 4.0:
            st.warning(f"RBC: {rbc} (Low) → Weak blood cells")
        elif rbc <= 5.5:
            st.success(f"RBC: {rbc} (Normal) → Healthy blood cells")
        else:
            st.error(f"RBC: {rbc} (High) → Possible issue")

    # WBC
    if wbc > 0:
        if wbc < 4500:
            st.warning(f"WBC: {wbc} (Low) → Weak immunity")
        elif wbc <= 11000:
            st.success(f"WBC: {wbc} (Normal) → Immune system ok")
        else:
            st.error(f"WBC: {wbc} (High) → Infection ho sakta hai")

    # Platelets
    if platelets > 0:
        if platelets < 150000:
            st.warning(f"Platelets: {platelets} (Low) → Dengue ya clotting issue")
        elif platelets <= 450000:
            st.success(f"Platelets: {platelets} (Normal) → Clotting normal")
        else:
            st.error(f"Platelets: {platelets} (High) → Risk of clotting")

    # -----------------------------
    # DETAILED EXPLANATION SECTION
    # -----------------------------
    st.subheader("📖 Detailed Explanation")

    st.write(f"""
**Hemoglobin (Hb) - {hb} g/dL**  
Yeh khoon mein oxygen le janay wali protein hai.  
Agar yeh 12 se kam ho → anemia ho sakta hai.

**RBC Count - {rbc} million/µL**  
Yeh surkh khoon ke khaliye hain jo oxygen transport karte hain.

**WBC Count - {wbc} cells/µL**  
Yeh immune system ka hissa hain. Infection mein barh jate hain.

**Platelets - {platelets} /cumm**  
Yeh khoon jamnay (clotting) mein madad karte hain.
    """)

    st.subheader("⚠️ Important Notes")
    st.write("""
- Low Platelets (<150,000) → Dengue ka risk  
- Low Hemoglobin (<12) → Iron deficiency  
- High WBC (>11,000) → Infection sign  
    """)

    # -----------------------------
    # BLOOD GROUP PREDICTION
    # -----------------------------
    if hb > 0 and rbc > 0 and wbc > 0 and platelets > 0:
        pred = model.predict([[hb, rbc, wbc, platelets]])
        st.subheader("🧬 Predicted Blood Group")
        st.success(pred[0])


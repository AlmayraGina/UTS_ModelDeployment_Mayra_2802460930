import streamlit as st
import joblib
import pandas as pd

model1 = joblib.load("artifacts/Salary_prediction_pipeline.pkl")
except Exception as e:
    st.text(traceback.format_exc())
model2 = joblib.load("artifacts/Placement_prediction_pipeline.pkl")

def main():

    st.title("Student Placement and Salary Prediction")

    gender = st.selectbox("Gender", ["Male", "Female"])
    extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Academic Values")
        ssc_percentage = st.slider("SSC Percentage", 0.0, 100.0, 70.0)
        hsc_percentage = st.slider("HSC Percentage", 0.0, 100.0, 70.0)
        entrance_exam_score = st.slider("Entrance Exam Score", 0.0, 100.0, 60.0)
        degree_percentage = st.slider("Degree Percentage", 0.0, 100.0, 70.0)
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
        attendance_percentage = st.slider("Attendance %", 0.0, 100.0, 80.0)

    st.sidebar.header("Skill Values")
    technical_skill_score = st.sidebar.slider("Technical Skill Score", 0.0, 100.0, 70.0)
    soft_skill_score = st.sidebar.slider("Soft Skill Score", 0.0, 100.0, 70.0)

    with col2:
        st.subheader("Experience Values")
        internship_count = st.number_input("Internships", 0, 10, 1)
        live_projects = st.number_input("Live Projects", 0, 10, 1)
        work_experience_months = st.slider("Work Experience (months)", 0, 60, 6)
        certifications = st.number_input("Certifications", 0, 20, 2)
        backlogs = st.number_input("Backlogs", 0, 10, 0)

    data = {
        "gender": gender,
        "ssc_percentage": ssc_percentage,
        "hsc_percentage": hsc_percentage,
        "degree_percentage": degree_percentage,
        "cgpa": cgpa,
        "entrance_exam_score": entrance_exam_score,
        "technical_skill_score": technical_skill_score,
        "soft_skill_score": soft_skill_score,
        "internship_count": internship_count,
        "live_projects": live_projects,
        "work_experience_months": work_experience_months,
        "certifications": certifications,
        "attendance_percentage": attendance_percentage,
        "backlogs": backlogs,
        "extracurricular_activities": extracurricular_activities
    }

    df = pd.DataFrame([data])

    if st.button("Predict"):
        prediction1 = model1.predict(df)[0]
        prediction2 = model2.predict(df)[0]

        st.subheader("Predicted Placement")
        if prediction2 == 1:
            st.success("Placed")
            st.subheader("Predicted Salary")
            st.success(f"Estimated Salary: {prediction1:.2f} LPA")
        else:
            st.error("Not placed")
            st.success(f"Estimated Salary: 0.0")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import pickle

# loading saved model
with open('career_model.pkl', 'rb') as f:
    my_model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    my_enc = pickle.load(f)

st.title("Career & Higher Studies Recommendation")
st.write("Enter your details and find out which career suits you best!")

st.header("Academic Scores")
maths = st.slider("Maths", 0, 100, 50)
science = st.slider("Science", 0, 100, 50)
arts = st.slider("Arts", 0, 100, 50)
commerce = st.slider("Commerce", 0, 100, 50)

st.header("Skills")
logic = 1 if st.checkbox("Good at logical thinking") else 0
creative = 1 if st.checkbox("I am creative") else 0
comm = 1 if st.checkbox("Good communication skills") else 0
tech = 1 if st.checkbox("Have technical skills") else 0

st.header("About You")
research = 1 if st.checkbox("I like research") else 0
helping = 1 if st.checkbox("I like helping people") else 0
business = 1 if st.checkbox("Interested in business") else 0
risk = 1 if st.checkbox("I take risks") else 0

if st.button("Find My Career"):
    inp = pd.DataFrame([{
        'math_score': maths,
        'science_score': science,
        'arts_score': arts,
        'commerce_score': commerce,
        'logical_thinking': logic,
        'creativity': creative,
        'communication': comm,
        'technical_skills': tech,
        'likes_research': research,
        'likes_helping': helping,
        'likes_business': business,
        'risk_taker': risk,
    }])

    out = my_model.predict(inp)
    career = my_enc.inverse_transform(out)[0]

    probs = my_model.predict_proba(inp)[0]
    table = pd.DataFrame({
        'Career': my_enc.classes_,
        'Confidence %': (probs * 100).round(1)
    }).sort_values('Confidence %', ascending=False)

    st.success("Recommended: " + career)
    st.subheader("All career matches:")
    st.dataframe(table)
import streamlit as st
import pandas as pd
import openai
import os
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import ollama
import data_info

# Load OpenAI API key

openai.api_key = data_info.open_ai_key
model = "gemma3:1b"

# Helper: Summarize DataFrame stats using LLM
def get_data_summary(df):
    sample_data = df.to_csv(index=False)
    schema = str(df.dtypes)
    prompt = f"""
    You are a data analyst. Analyze the following dataset and summarize key insights, like:
    - Number of rows and columns
    - Data types
    - Missing values
    - Descriptive statistics
    - Anything noteworthy

    Schema:
    {schema}

    Sample Data:
    {sample_data}
    """
    try:
        result = ollama.generate(model=model, prompt=prompt)
        return result.response

    except Exception as e:
        return f"OpenAI error: {e}"


# Helper: Validate the analysis with LLM
def verify_summary(df, summary):
    prompt = f"""Check if the following summary for the given database has any inconsistencies, hallucinations or errors. If it looks correct, say 'Valid'. Else explain the issue.
    Database:{df}

    Summary:
    {summary}"""
    try:
        result = ollama.generate(model=model, prompt=prompt)
        return result.response
    except Exception as e:
        return f"OpenAI error: {e}"


# Streamlit UI
st.title("📊 Data Analyst Agent")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        with st.spinner("Generating summary using AI..."):
            summary = get_data_summary(df)
            validation = verify_summary(df, summary)

            # Retry once if validation failed
            if "valid" not in validation.lower():
                st.warning("Initial summary validation failed. Retrying...")
                summary = get_data_summary(df)
                validation = verify_summary(summary)

        st.subheader("🔍 Summary of the Dataset")
        st.text(summary)
        st.caption(f"Validation Result: {validation}")

        # Optional: show some plots if numeric data exists
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_cols:
            st.subheader("📈 Auto-Generated Charts")
            for col in numeric_cols[:3]:  # Show up to 3 plots
                st.markdown(f"**Distribution of `{col}`**")
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading or analyzing file: {traceback.format_exc()}")

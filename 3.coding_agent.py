import streamlit as st
import traceback
import contextlib
import io
from openai import OpenAI
import data_info

# --- Function to call OpenAI API ---
def generate_code_from_description(user_description):

    prompt = f"""
    You are a helpful assistant that writes clean, testable Python code based on a user description.

    The generated code will be executed using `exec(code, )` inside a Streamlit app. 
    Hence, make sure:
    - All relevant results are shown using `print(...)` statements
    - You do not include any explanations or markdown syntax like ```python or ```
    - Return **only the Python code** — no headings, comments, or intro text

    User's description:
    {user_description}
    """

    try:
        client = OpenAI(api_key=data_info.open_ai_key)
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0,
        )
        return response.output_text
    except Exception as e:
        return f"OpenAI error: {e}"


def generate_code_from_description_optimize(feedback,code):
    prompt = f"""You are a helpful assistant that writes clean, testable Python code given a description. You generated below code and
    critique has suggested some improvements. Your task in now to rewrite the code basis on the feedback
            Generated Code:
            {code}
            Critique Feedback:
            {feedback}
            Generate the new code and do not add any explanation. Return only plain code, remove anything like ```python or ```.
              The generated code will be executed using `exec(code, )` inside a Streamlit app. 
                Hence, make sure:
                - All relevant results are shown using `print(...)` statements
                - You do not include any explanations or markdown syntax like ```python or ```
                - Return **only the Python code** — no headings, comments, or intro text
             """
    try:
        client = OpenAI(api_key=data_info.open_ai_key)
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0,
        )
        return response.output_text
    except Exception as e:
        return f"OpenAI error: {e}"

# --- Function to run code ---
def execute_python_code(code):
    output = io.StringIO()
    print("Inside execute_python_code")
    print(code)
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {})
        print("printing output")
        print("========"+output.getvalue()+"========")
        return output.getvalue()
    except Exception as e:
        print("Inside exception of execute_python_code")
        print(e)
        return traceback.format_exc()

# --- Function to critique code ---
def critique_code(code):
    critique_prompt = f"""Analyze and critique the following Python code. 
        Point out any:
        - performance or logic issues
        - edge cases not handled
        - input validation or error handling improvements
        - stylistic or best-practice violations
        
        Also mention if the code could fail in certain scenarios.
        
        Code:
        {code}
        """
    try:
        client = OpenAI(api_key=data_info.open_ai_key)
        response = client.responses.create(
            model="gpt-4o-mini",
            input=critique_prompt,
            temperature=0,
        )
        return response.output_text
    except Exception as e:
        return f"Critique error: {e}"

# --- Streamlit UI ---
st.title("=====Python Coding Agent======")

code_description = st.text_area("Describe the code you want to generate:", height=200)
run_button = st.button("Generate, Run, and Critique")

if run_button and code_description:
    with st.spinner("Generating code..."):
        gen_prompt = f"Write a complete Python function based on the following description. Include test cases:\n\n{code_description}"
        generated_code = generate_code_from_description(gen_prompt)
        print(generated_code)

    st.subheader("===Generated Code====")
    st.code(generated_code, language='python')

    with st.spinner("Executing code..."):
        execution_result = execute_python_code(generated_code)
        print(execution_result)

    st.subheader("===Execution Output====")
    st.text(execution_result)

    # If code failed, retry
    final_code = generated_code
    if "Traceback" in execution_result:
        st.warning("== Execution failed. Retrying without test cases...")
        retry_prompt = f"Write only the main function (no test cases) for this description:\n\n{code_description}"
        retry_code = generate_code_from_description(retry_prompt)
        retry_result = execute_python_code(retry_code)

        st.subheader(" Retry Code")
        st.code(retry_code, language='python')
        st.subheader(" Retry Output")
        st.text(retry_result)

        final_code = retry_code
        execution_result = retry_result

    # Run critique on whichever version succeeded
    with st.spinner("Critiquing code..."):
        critique = critique_code(final_code)
        st.subheader("===== Code Critique =====")
        st.text(critique)
        final_result = generate_code_from_description_optimize(gen_prompt, critique)
        st.subheader(" Retry Code")
        st.code(final_result, language='python')
        retry_result = execute_python_code(final_result)
        st.subheader(" Retry Output")
        st.text(retry_result)
        print(final_result)



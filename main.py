from backend.core import run_llm
import streamlit as st
from streamlit_chat import message


def create_sources_string(source_urls: set) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()

    source_string = "Sources: \n"

    for i, source in enumerate(sources_list):
        source_string += f"{i + 1}.{source}\n"
    return source_string


st.header("Langchain - Documentation Assistant")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_response_history" not in st.session_state:
    st.session_state["chat_response_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if prompt:
    with st.spinner("Generating Response..."):
        generated_response = run_llm(prompt, chat_history=st.session_state["chat_history"])
        sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])
        formatted_response = f"{generated_response['answer']}\n\n {create_sources_string(sources)}"

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_response_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))


if st.session_state["chat_response_history"]:
    for response, prompt in zip(st.session_state["chat_response_history"], st.session_state["user_prompt_history"]):
        message(prompt, is_user=True)
        message(response)

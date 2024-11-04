from typing import Set
import streamlit as st
from backend.core import run_query
from streamlit_chat import message

st.header("Demo Bot")

prompt = st.text_input("prompt",placeholder="Enter your prompt here...")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
    
if "chat_answer_history" not in st.session_state:
       st.session_state["chat_answer_history"] = []

def create_sources_string(sources : Set[str]) -> str:
    if not sources:
        return ""
    sources_list = list(sources)
    sources_list.sort()
    sources_string = "sources \n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}"
    return sources_string
    

if prompt:
    with st.spinner("Generating response..."):
        print(prompt)
        generated_response = run_query(query=prompt)
        sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])        
        formatted_response = f" {generated_response['result']} \n\n {create_sources_string(sources)} "
        print(formatted_response)
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        
if st.session_state["chat_answer_history"]:
    for generated_response , userQuery in zip(st.session_state["user_prompt_history"], st.session_state["chat_answer_history"]):
        message(generated_response, is_user=False)
        message(userQuery, is_user=True)
import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug

set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

title_prompt= PromptTemplate(
    input_variable = ["topic"],
    template = """  You are an experienced speech writer.
    You need to craft an impactful title for a speech
    on the following topic: {topic}
    Answer exactly with one title."""
    )

speech_prompt= PromptTemplate(
    input_variable = ["title"],
    template = """  You need to write a powerful speech of 500 words
    for the following title: {title}"""
    )

first_chain = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title),title)[1])

second_chain = speech_prompt | llm

final_chain = first_chain | second_chain

st.title("Speech Generator")

topic = st.text_input("Enter a topic")

if topic:
    response = final_chain.invoke({"topic":topic})
    st.write(response.content)

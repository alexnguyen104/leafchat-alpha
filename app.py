# langchain
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# streamlit
import streamlit as st

# load env values
import os
from dotenv import load_dotenv

# load_dotenv()
# OPENAI_API_KEY = os.getenv('OPENAI-API-KEY')

OPENAI_API_KEY = st.secrets['OPENAI-API-KEY']

from storage_his import storage_his
storage = storage_his

def generate_response(input_text):
    model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
            "You are only allowed to answer questions that are related to Vietnam. If other questions are asked, please politely reject it."
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    )


    def limit_history(messages, k = 3):
        return messages[-k:]

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in storage:
            storage[session_id] = ChatMessageHistory()
        return storage[session_id]

    chain = (
        RunnablePassthrough.assign(messages=lambda x: limit_history(x["messages"]))
        | prompt
        | model
    )

    chain_with_history = RunnableWithMessageHistory(chain, get_session_history=get_session_history, input_messages_key="messages")

    config = {"configurable": {"session_id": "abc20"}}
    return chain_with_history.stream({"messages": [HumanMessage(content=input_text)]}, config=config)
    # return "hi"


st.title('Leafchat (alpha)')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Chat with Leafchat"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": prompt})



# for i in chain_with_history.stream({"messages": [HumanMessage("my name is alex")]}, config=config):
#     print(i.content, end='', flush=True)

# model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system",
#         "You are only allowed to answer questions that are related to Vietnam. If other questions are asked, please politely reject it.",
#         ),
#         MessagesPlaceholder(variable_name="messages")
#     ]
# )

# storage = {}

# def limit_history(messages, k = 3):
#     return messages[-k:]

# def get_session_history(id: str) -> BaseChatMessageHistory:
#     if id not in storage:
#         storage[id] = ChatMessageHistory()
#     return storage[id]

# chain = (
#     RunnablePassthrough.assign(messages=lambda x: limit_history(x["messages"]))
#     | prompt
#     | model
# )

# chain_with_history = RunnableWithMessageHistory(chain, get_session_history=get_session_history, input_messages_key="messages")

# config = {"configurable": {"session_id": "abc20"}}
# res = chain_with_history.invoke({"messages": [HumanMessage("hi")]}, config=config).content

# print(res)

import streamlit as st
from dotenv import load_dotenv
import os
import uuid

from llama_index.llms.bedrock import Bedrock
from llama_index.core import ServiceContext, StorageContext
from llama_index.embeddings.bedrock import BedrockEmbedding

# enabling debug
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager

# exceptions
from botocore.exceptions import ClientError

# import the object for message
from llama_index.core.llms import ChatMessage, MessageRole
import json
load_dotenv()

## helper function to convert messages to chat messages
def convert_messages(messages):
    converted_messages = []
    for message in messages:
        role = message['role']
        content = message['content']
        if role == "assisstant":
            role = MessageRole.ASSISTANT
        elif role=="system":
            role = MessageRole.SYSTEM
        else:
            role = MessageRole.USER
        converted_messages.append(ChatMessage(role = role, content = content))
    return converted_messages
        


# initialize the service and storage context
debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[debug])
llm = Bedrock(
    model=os.environ["BEDROCK_MODEL"],
    temperature=os.environ["CHAT_TEMPERATURE"],
    profile_name=os.environ["AWS_PROFILE_NAME"],
)
embedding = BedrockEmbedding.from_credentials(
    aws_profile=os.environ["AWS_PROFILE_NAME"]
)
# create a service context using the Bedrock embeddings
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embedding  # , callback_manager=callback_manager
)

st.set_page_config(
    page_title="Chat with Claude",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
    
st.title("💬 Chat with Claude 💬")

# add a sidebar for chatting with PDF
with st.sidebar:
    st.page_link("main.py", label="Home", icon="🏠")
    st.page_link("pages/pdf.py", label="PDF", icon="📁")
    st.page_link("pages/website.py", label="Website", icon="🌐")
    st.page_link("pages/chat.py", label="Chat", icon="💬")
    
# create 2 tabs. one option is to set the system message for the change
chat_tab, system_tab = st.tabs(['Chat','Settings'])

with chat_tab:
    st.subheader("Chat")
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "I am a helpful assistant. Ask me a question..",
            }
        ]

    # check and initialize index
    if "chat_engine" not in st.session_state.keys():
        st.session_state.chat_engine = llm

    if prompt := st.chat_input("Your question: "):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        # skip the system messages
        if message['role'] != 'system':
            with st.chat_message(message["role"]):
                st.write(message["content"])
                # call the chat engine with user prompt.. variable prompt is defined earlier
    
    # if the last message is from a user, now, run the llm    
    if (st.session_state.messages[-1])["role"] == "user":
        with st.spinner('Waiting for response from GPT...'):
            try:
                response = st.session_state.chat_engine.chat(messages = convert_messages(st.session_state.messages))                
                response = json.loads(response.json())['message']['content']
                with st.chat_message('assistant'):
                    st.write(response) 

                # add the message into the message history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )
            except ClientError as e:
                st.error(f"ClientError: {e}")

with system_tab:
    # let the user set a system prompt
    system_prompt = st.text_area(label='System Prompt',
                                 help="Optional System Prompt to change the behavior of the chat"                                 
    )
    if st.button('Update'):
        if system_prompt != '' and system_prompt!=' ':
            # wipe out the history and reset the messages
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {            
                    "role": "assistant",
                    "content": "I am a helpful assistant. Ask me a question..",
                }                
            ]
            st.session_state.messages = messages
            st.success('System prompt updated successfull!')
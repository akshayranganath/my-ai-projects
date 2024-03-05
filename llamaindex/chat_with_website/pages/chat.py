import streamlit as st
# import the utils functions
from uiutils import build_page_structure, handle_chat, initialize
import json

initialize()
build_page_structure("Chat with Bedrock", "💬")
    
# create 2 tabs. one option is to set the system message for the change
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
#chat_tab, system_tab = st.tabs(['Chat','Settings'])

#with chat_tab:
handle_chat( clear_session=False )

#with system_tab:

import chainlit as cl
from caption_with_lava import get_image_caption
from chainlit.types import ThreadDict


@cl.on_chat_start
def on_chat_start():
    print("A new chat session has started!")


@cl.on_message
async def on_message(msg: cl.Message):
    #print("The user sent: ", msg.content)
    print(msg.elements[0].path)
    caption = get_image_caption(msg.elements[0].path,query=msg.content)
    print(caption)
    await cl.Message(f'{caption}').send()

@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")


@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    print("The user resumed a previous chat session!")

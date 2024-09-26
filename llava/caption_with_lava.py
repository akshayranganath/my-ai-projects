from llama_index.llms import Ollama
import base64
from llama_index.schema import ImageDocument
from llama_index.llms import ChatMessage

# we require a base 64 encoded string of the image bytes - this function helps out.
def get_b64_image(image_file:str)->str:    
    with open(image_file,'rb') as f:
        data = f.read()
    b64_image = base64.b64encode(data).decode()
    return b64_image


def get_image_caption1(image_path:str, query='Can you desribe this image?')->str:
    # load the llava model with Ollama interface. I am keeping a very high timeout to handle larger images
    print(query)
    llm = Ollama(model='llava', temperature=0, request_timeout=300.0,verbose=True)    
    b64_image = get_b64_image(image_path)    
    llava_response = llm.complete(
        query,
        images=[b64_image],
    )
    return llava_response

def get_image_caption(image_path:str, query='Can you desribe this image?')->str:
    # load the llava model with Ollama interface. I am keeping a very high timeout to handle larger images
    
    llm = Ollama(model='llava', temperature=0, request_timeout=300.0,verbose=True)            
    b64_image = get_b64_image(image_path)
    
    messages = [
        ChatMessage(
            role="system", content="You are an assistant who can analyze and extract details from images."
        ),
        ChatMessage(role="user", content=query, images=[b64_image]),
    ]
    llava_response = llm.chat(messages)
    return llava_response

if __name__ == '__main__':
    print(get_image_caption('/Users/akshayranganath/Downloads/nike-test.jpg'))
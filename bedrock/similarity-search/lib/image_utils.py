from PIL import Image
import requests
from uuid import uuid1
import tempfile
import os

def download_and_save_image(url:str)->str:
    result = None
    try:
        img = Image.open(requests.get(url, stream=True).raw)
        file_name = f'{tempfile.gettempdir()}/{uuid1()}.{img.format}'
        #print(file_name)
        #print(f'format={img.format}')
        img.save(file_name)#, img.format)
        result = file_name
    except Exception as e:
        print(e)
    return result

def delete_image(file_path:str)->bool:
    try:
        os.remove(file_path)
    except Exception as e:
        print(e)
    return True
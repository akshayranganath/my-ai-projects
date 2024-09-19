from mybedrockobject import MyBedrockObject
from utils import load_data_from_url
import csv

bedrock_obj = MyBedrockObject()
urls = []
with open('./data/test.txt') as f:
    reader = csv.reader(f)
    
    for url in reader:
        urls.append(url[0])

print(f'Trying to bulk import : {len(urls)} URLs')        
i = 0
for url in urls:    
    load_data_from_url(url, bedrock_obj=bedrock_obj, show_progress=False)
    i+= 1    
    print(f'{i} urls processed..{url}')
print('Completed the bulk import.')            
import re

md = """
---
layout: post
title: Pretty Print JSON &#38; Move it to Command Line
comment: true
description: Pretty printing JSON is a common operation. I show how to get this done and set it as a command.
image: https://akshayranganath-res.cloudinary.com/image/upload/f_auto,q_auto/blog/command_prompt.png
tag: [json,pretty-print]
---
Pretty printing JSON is a very common operation. In this post, I show how to build the code and make the script an executable as a first-class command.
"""

def get_metadata(markdown):

    FRONT_MATTER_RE = re.compile(
        r"^\s*\ufeff?---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
        re.DOTALL
    )

    m = FRONT_MATTER_RE.match(markdown)
    meta_block = m.group(1)
    body = markdown[m.end():]

    title = description = image = None
    tags = []

    for line in meta_block.splitlines():
        if line.startswith('title'):
            title = line.split('title:')[1].strip()
        elif line.startswith('description'):
            description = line.split('description:')[1].strip()
        elif line.startswith('image'):
            image = line.split('image:')[1].strip()
        elif line.startswith('tag'):
            tags = line.split('tag:')[1].strip()
            if tags:
                tags = tags.split('[')[1].split(']')[0].split(',')
    print(title, description, image, tags)            
    metadata = {
        "title": title,
        "description": description,
        "image": image,
        "tags": tags
    }
    return (metadata, body)

if __name__=="__main__":
    get_metadata(md)
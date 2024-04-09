# Image Similarity Search

I needed to create a demo to show the use of **multimodal** embedding and also help implement a solution where a user could upload an image and search for similar images. To accomplish both, I have implemented this project.

## Architecture Details

I began by using Amazon Bedrock and `Titan Embed Image` to create the embeddings. However, I realized this was more complex than necessary. ChromaDB[https://docs.trychroma.com/] implements [multimodal embedding](https://docs.trychroma.com/multi-modal). I used this functionality for the project.

## How does it work?

**Assumptions**

* Images are uploaded to Cloudinary
* Streamlit has write access to the system's temp directory
* ChromaDB's embedding is based on [OpenClip](https://huggingface.co/docs/hub/open_clip). 
* Default similarity search is eucleadean distance `l2` algorithm


### Indexing

When adding an image to the index, the following happens:

* User enters an image URL.
* The image is downloaded to temp folder.
* Image is embedded using Chroma's multi-modal algorithm. Index metadata includes the image URL.
* Downloaded image from temp folder is deleted.

### Searching

When a user wants to search for an image, the following occurs:

* User enters an image URL.
* The image is downloaded to temp folder.
* Search query is executed with the downloaded image. 

Internally, chroma embeds the image and then runs a search for the match. The query returns the N-top results (defaults to 3). On the front-end, the results are displayed only if the distance between the input image and result is less than 0.5. This is to filter out results that are too far apart.

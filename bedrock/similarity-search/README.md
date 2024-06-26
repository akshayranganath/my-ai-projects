# Image Similarity Search

I needed to create a demo to show the use of **multimodal** embedding and also help implement a solution where a user could upload an image and search for similar images. To accomplish both, I have implemented this project.

![search result](https://res.cloudinary.com/dbmataac4/image/upload/f_auto,q_auto,h_550,e_sharpen/workshop/presentation/Screenshot_2024-04-18_at_10.30.57_AM.png)

## Architecture Details

ChromaDB[https://docs.trychroma.com/] implements [multimodal embedding](https://docs.trychroma.com/multi-modal). I used this functionality for the project.

## How does it work?

**Assumptions**

* Images are accessible as public URLs.
* Streamlit has write access to the system's temp directory
* ChromaDB's embedding is based on [OpenClip](https://huggingface.co/docs/hub/open_clip). 
* Default similarity search is Eucleadean distance `l2` algorithm. However, in this project, I am using `cosine distance` metric.
* Lastly, the search looks for the 3 closes matches and a cosine distance less than 0.5.

For more details on using ChromaDB and the algorithms, please refer to this excellent article, **[Embeddings and Vector Databases With ChromaDB – Real Python](https://realpython.com/chromadb-vector-database/)**.


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

## Demo

Here's a demo of using the application: [video](https://res.cloudinary.com/dbmataac4/video/upload/f_auto,q_auto/workshop/visually_similar_search.mp4).
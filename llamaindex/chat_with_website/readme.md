# Chat with Website

## RAG Application using Bedrock

This application provides the ability to perform Retrieval Augmented Generation (RAG) and chat with AWS provided models under the Bedrock offering. Specifically, this app allows the following:

* Read and digest data from a web page
* Ingest and index data from a PDF
* Allow users to Chat with their data

## Setup
### Install

First install all the dependecies. You'll require python 3.x to run the code. 

```
pip install -r requirements.txt
```

### Execute

To start the application, run the `streamlit` command. The application starts at a default port of `8501`.

You may optionally want to execute the [screen](https://www.howtogeek.com/662422/how-to-use-linuxs-screen-command/) command so that the application is not tied to your terminal access.

```
streamlit run main.py
```

If you want to make changes to the UI like changing the theme or server port, update the `/.streamlit/config.toml` file. For more information, please refer to [streamlit docs](https://docs.streamlit.io/library/advanced-features/configuration).

### Installing on EC2

If you are planning to run this application from an Amazon EC2 instance, you'll need to do the additional things:

* Create an IAM role. For this role, provide access to Bedrock. There is a pre-defined policy to `BedrockFullAccess` to help get you started. My suggestion is that create your own role that provides just enough permission necessary.
* Update the security group to make 2 changes:
    * For your EC2 instance, apply the IAM role created above.
    * Update the Inbound rules to allow access on port `8501`.
* Follow the install/execute steps above. 

## Architecture

This application relies on the following technologies:

* [Streamlit](https://docs.streamlit.io/get-started): UI Application layer
* [Llama-Index](https://docs.llamaindex.ai/en/stable/): Library providing the interface to talk to LLM and embedding models.
* [Amazon Bedrock](https://aws.amazon.com/bedrock/): Provides the actual LLM and Embedding model.
* [ChromDB](https://docs.trychroma.com/): (local) Vector database for storing the indexed data.

For this project, I am using the following models:

* `anthropic.claude-instant-v1`: LLM powering the application
* `amazon.titan-embed-g1-text-02`: Providing the embedding services


### Application structure

The application consists of a very simple setup:

* A page to help users add their websites or PDF for embedding.
* Chat interface to talk to the documents.

### Application flow

The application flow can be broken down into 2 primary use cases - 

* loading data
* chatting with data

**Loading Data**

For loading website data, I am using `SimpleWebPageReader` for reading website. For loading and reading PDFs, I am using `PyPDF2`. Currently, I am only reading the text content. Once the data is read, it is split into chunks. I am relying on the default chunk size and overlap.

Once the document is chunked, it is sent for embedding. For this, I am using [Amazon Titan's](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html) text embedding model. It does support other modalities and I may change it in the future.

**Chatting with Data**

Once data is ingested, embedded and indexed, we can chat! On the chat interface, I provide an option to modify the system prompt. Without any specific tweaks, the default chat mode starts with the message `I am a helpful assistant. Ask me a question`.

When a user types a chat query, I check the data in ChromaDB. If any match is found it is included in the prompt that is passed onto to the LLM. If nothing is found, it passes on just the prompt. So the chat can act either as data-specific or a generic chat.

[Anthropic's Claude](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html#claude-messages-supported-models) is the base LLM. Specifically, I am using _Claude Instant_ which is a lighter version of the LLM base model.

## Next Steps

This is just a training for myself to understand how the different technologies fit together. For the next steps, I may add the following:

* provide multi-modal embedding and chat capability
* saving the results
* switching to a database like [Pinecone](https://aws.amazon.com/marketplace/pp/prodview-xhgyscinlz4jk)
* lastly explore a serverless alternative for running the backend tasks.
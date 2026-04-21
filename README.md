# RAG System
A simple Retrieval-Augmented Generation (RAG) system that combines document embeddings with Azure OpenAI to answer questions based on provided documents without using LangChain.

## Features

- **Document Chunking**: Splits documents into overlapping chunks for processing
- **Embeddings**: Uses Azure OpenAI's text-embedding-ada-002 model for generating embeddings
- **Vector Storage**: Stores embeddings in PostgreSQL for similarity search
- **Question Answering**: Retrieves relevant document chunks and uses Azure OpenAI's chat model to generate context-aware answers

## Prerequisites

- Python 3.8+
- PostgreSQL database
- Azure OpenAI API access with:
  - API Key
  - Endpoint URL
  - Deployment name
  - API version

## Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install openai psycopg2-binary numpy
   ```

3. **Configure environment variables**
   - Copy `.env.example` to `.env`
   - Fill in your Azure OpenAI credentials:
     ```
     AZURE_OPENAI_ENDPOINT=<your-endpoint>
     AZURE_OPENAI_API_KEY=<your-api-key>
     AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
     AZURE_OPENAI_API_VERSION=<your-api-version>
     ```

4. **Setup PostgreSQL database**
   - Create a database called `prepareisio`
   - Create the embeddings table:
     ```sql
     CREATE TABLE document_embeddings (
         id SERIAL PRIMARY KEY,
         chunk TEXT UNIQUE NOT NULL,
         embedding VECTOR(1536)
     );
     ```

## Usage

Place your document in `document.txt`, then run:

```bash
python rag.py
```

The script will:
1. Load and chunk the document
2. Generate embeddings for each chunk
3. Store embeddings in PostgreSQL
4. Prompt you for a question
5. Retrieve relevant chunks
6. Generate an answer using Azure OpenAI

## Project Structure

- `rag.py` - Main RAG implementation
- `document.txt` - Document to embed and search
- `.env.example` - Environment variables template

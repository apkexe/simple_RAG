from openai import AzureOpenAI
import os 
import psycopg2
import numpy as np

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

llm = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT,
    api_version=AZURE_API_VERSION
)


def embed_text(text):
    response = llm.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def load_document(filepath):
    with open(filepath, 'r') as file:
        return file.read()


def chunk_document(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap 
    return chunks

def upsert_embedding_to_db(chunk, embedding):
    conn = psycopg2.connect(
        dbname="prepareisio",
        user="admin",
        password="admin",
        host="localhost"
    )
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO document_embeddings (chunk, embedding)
        VALUES (%s, %s)
        ON CONFLICT (chunk) DO UPDATE SET embedding = EXCLUDED.embedding
    """, (chunk, embedding))
    conn.commit()
    cursor.close()
    conn.close()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve (query_embedding, top_k=5):
    conn = psycopg2.connect(
        dbname="prepareisio",
        user="postgres",
        password="postgres",
        host="localhost"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT chunk, embedding FROM document_embeddings")
    similarities = []
    for chunk, embedding in cursor.fetchall():
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity)) 
    
    similarities.sort(key=lambda x: x[1], reverse=True)   
    return similarities[:top_k]

if __name__ == "__main__":
    document = load_document('document.txt')
    chunks = chunk_document(document)
    embeddings = [embed_text(chunk) for chunk in chunks]
    for chunk, embedding in zip(chunks, embeddings):
        upsert_embedding_to_db(chunk, embedding)

    # Implement retrieval and question-answering logic here
    input_query = input("Enter your question: ")
    retrieved_chunks = retrieve(embed_text(input_query))
    print ("Retrieved Chunks:")
    for chunk, similarity in retrieved_chunks:
        print(f" - (similarity: {similarity:.4f}) Chunk: --- {chunk}\n")
    
    context = "\n\n".join([chunk for chunk, _ in retrieved_chunks])

    prompt = f"""Answer the question based on the following context. If the answer is not contained within the context, say you don't know.
            Context: {context}
            Question: {input_query}
            \n\n"""
    
    completion = llm.chat.completions.create(
        model = AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
            ]
        )
    
    print(completion.choices[0].message.content)

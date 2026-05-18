import chromadb
from sentence_transformers import SentenceTransformer, util

embedding_model=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_files_only=True)

client= chromadb.PersistentClient(path="test_db2")
collection=client.get_or_create_collection("conversations")

#text= "we fixed a python bug"
#embedding= embedding_model.encode(text).tolist()

conversations = [
    {"id": "convo_1", "text": "User: we fixed a python bug\nAI: glad we sorted that out"},
    {"id": "convo_2", "text": "User: what should i eat today\nAI: maybe try biryani"},
    {"id": "convo_3", "text": "User: explain neural networks\nAI: they are layers of math"},
    {"id": "convo_4", "text": "User: my python code has an error\nAI: lets debug it together"},
    {"id": "convo_5", "text": "User: i love playing chess\nAI: chess is a great game"},
]

for convo in conversations:
    embedding= embedding_model.encode(convo ["text"]).tolist()
    collection.add(
        ids=[convo["id"]],
        embeddings=[embedding],
        documents=[convo["text"]]
        )
    
print("stored:",collection.count(),"conversations")

query="i have a bug in my code"
query_embedding=embedding_model.encode(query).tolist()

results=collection.query(
    query_embeddings=[query_embedding],
    n_results=2
    )

for doc in results ["documents"][0]:
    print(doc)
    print("---")
    
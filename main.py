from agents.indexer.indexer import (
    load_documents,
    clean_documents,
    split_into_chunks,
    create_faiss_index,
    load_faiss_index,
    search_in_faiss
)

from dotenv import load_dotenv
load_dotenv()

# Crear índice (si no existe)
docs = load_documents("data/docs_iniciales")
docs = clean_documents(docs)
chunks = split_into_chunks(docs)
create_faiss_index(chunks, "data/faiss_index")

# Cargar índice
vector_store = load_faiss_index("data/faiss_index")

# Probar búsqueda
query = "De qué trata este documento?"
resultados = search_in_faiss(vector_store, query, k=2)

print("Resultados:")
for r in resultados:
    print("-----")
    print(r.page_content)
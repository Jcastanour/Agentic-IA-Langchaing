from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os


def load_documents(folder_path):
    """
    Carga todos los PDF que encuentre en la carpeta indicada
    y devuelve una lista de los documentos que hay.
    """
    loader = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documentos = loader.load()
    return documentos

def clean_documents(docs):
    """
    Limpia texto básico:
    - eliminar los espacios dobles
    - eliminar los saltos de línea repetidos
    """

    documentos_limpios = []

    for doc in docs:
        texto = doc.page_content

        # Limpieza básica
        texto = texto.replace("  ", " ")
        texto = texto.replace("\n\n", "\n")
        texto = texto.strip()

        nuevo = type(doc)(
            page_content=texto,
            metadata=doc.metadata # Informacion util sobre el documento
        )

        documentos_limpios.append(nuevo)

    return documentos_limpios

def split_into_chunks(docs):
    """
    Divide los documentos en pedazos (chunks) de tamaño fijo.
    Cada chunk tendrá 600 caracteres con 100 de solapamiento pa que no queden cortado.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    return chunks

def create_faiss_index(chunks, index_folder):
    """
    Crea un índice FAISS a partir de los chunks.
    Convierte cada chunk en un embedding y guarda el índice localmente.
    """

    # 1. Inicializar el modelo de embeddings de Google
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    # 2. Crear la base vectorial desde los chunks
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 3. Crear carpeta si no existe
    os.makedirs(index_folder, exist_ok=True)

    # 4. Guardar el índice (dos archivos: index.faiss y index.pkl)
    vector_store.save_local(index_folder)

    print("Índice FAISS guardado en:", index_folder)

def load_faiss_index(index_folder):
    """
    Carga el índice FAISS desde la carpeta especificada.
    Debe usar el mismo modelo de embeddings que se usó al crearlo.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    vector_store = FAISS.load_local(
        index_folder,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store

def search_in_faiss(vector_store, query, k=3):
    """
    Busca en el índice FAISS usando la consulta (query).
    Retorna los k chunks más similares.
    """
    resultados = vector_store.similarity_search(query, k=k)
    return resultados
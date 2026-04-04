# Load PDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# ✅ Fix path (use forward slash)
loader = PyPDFLoader("documnet_Loader/hospital_charges_aiims.pdf")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# ✅ Use Mistral embeddings
embedding_model = MistralAIEmbeddings(model="mistral-embed")

# Store in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

print("✅ Vector DB created successfully")
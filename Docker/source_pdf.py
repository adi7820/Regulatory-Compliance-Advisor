from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import dotenv
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import time


dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
api_key = "9610b4a7-fd58-4884-8463-b4861abb8535"

def update_vectorstore_with_pdf(index_name, file_path):
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    embeddings = OpenAIEmbeddings()

    # file_path = "pdf_data/health/questions-answers-implementation-medical-devices-vitro-diagnostic-medical-devices-regulations-eu-2017-745-eu-2017-746_en.pdf"
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    # print(pages[1].metadata)

    documents = []
    documents.extend(pages)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    start_time = time.time()

    vectorstore_from_docs = PineconeVectorStore.from_documents(
            texts,
            index_name=index_name,
            embedding=embeddings
        )

    time_taken = time.time() - start_time
    source = Path(file_path).name

    return 'Source: ' + source + '\n\n' + 'Time Taken To Store in VectorDB: ' + str(f"{time_taken:.2f}") + 'sec'
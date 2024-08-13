from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
import os
import dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import time


dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
api_key = "9610b4a7-fd58-4884-8463-b4861abb8535"

def update_vectorestore_with_url(index_name, inp):
    pc = Pinecone(api_key=api_key)

    index = pc.Index(index_name)
    # index.describe_index_stats()

    embeddings = OpenAIEmbeddings()
    # vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    urls = []
    urls.append(inp)

    loader = UnstructuredURLLoader(urls=urls, show_progress_bar=True)
    urls = loader.load()

    documents = []
    documents.extend(urls)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    start_time = time.time()

    vectorstore_from_docs = PineconeVectorStore.from_documents(
            texts,
            index_name=index_name,
            embedding=embeddings
        )

    time_taken = time.time() - start_time
    source = documents[0].metadata['source']

    return 'Source: ' + source + '\n\n' + 'Time Taken to Store in VectorDB: ' + str(f"{time_taken:.2f}") + 'sec'
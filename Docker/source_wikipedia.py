from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores.pinecone import Pinecone
import os
import dotenv
from langchain_community.document_loaders import WikipediaLoader
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import time

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
api_key = "9610b4a7-fd58-4884-8463-b4861abb8535"

def update_vectorstore_with_query(query, index_name):
    docs = WikipediaLoader(query=query, doc_content_chars_max = 100000, load_max_docs=1).load()
    # print(docs[0].metadata)
    documents = []
    documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    # # configure client
    pc = Pinecone(api_key=api_key)

    index = pc.Index(index_name)
    # print(index.describe_index_stats())

    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    start_time = time.time()

    vectorstore_from_docs = PineconeVectorStore.from_documents(
            texts,
            index_name=index_name,
            embedding=embeddings
        )

    time_taken = time.time() - start_time
    title = docs[0].metadata['title']
    summary = docs[0].metadata['summary']
    source = docs[0].metadata['source']

    return 'Title: ' + title + '\n\n' + 'Summary: ' +  summary + '\n\n' + 'Source: ' + source + '\n\n' + 'Time Taken To Store in VectorDB: ' + str(f"{time_taken:.2f}") + 'sec'
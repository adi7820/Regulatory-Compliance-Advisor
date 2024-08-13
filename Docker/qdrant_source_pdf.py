from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from cleantext import clean
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import dotenv
import os
import time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
dotenv.load_dotenv()

qdrant_api_key = "7hCtBhv8qcjRZ6O6DdeEIvd_lXkTombKa64-MrO6nsdLptTWecPFtQ"
hf_token = "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"

def update_qdrant_vectorstore_with_pdf(file_path, collection_name):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    raw_text =pages[0].page_content

    clean_text = clean(text=raw_text,
                    fix_unicode=True,
                    to_ascii=True,
                    lower=True,
                    no_line_breaks=False,
                    no_urls=False,
                    no_emails=False,
                    no_phone_numbers=False,
                    no_numbers=False,
                    no_digits=False,
                    no_currency_symbols=False,
                    no_punct=False,
                    lang="en"
                    )
    
    metadata = {
            "source": pages[0].metadata['source'],
        }

    custom_doc = Document(page_content=clean_text, metadata=metadata)
    document =[custom_doc]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts = text_splitter.split_documents(document)

    model_name = 'dunzhang/stella_en_400M_v5'

    # Define model parameters
    model_kwargs = {'device': 'cuda', 'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': True}

    # Initialize HuggingFace embeddings
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)
    # Define the URL for Qdrant
    url="https://a1fbbfce-ca47-437d-b308-e79724650692.us-east4-0.gcp.cloud.qdrant.io:6333"

    # Initialize Qdrant client
    client =  QdrantClient(url=url, api_key=qdrant_api_key)

    start =time.time()
    # Insert documents into the collection
    qdrant = Qdrant.from_documents(texts,embeddings_model,url=url,prefer_grpc=False,api_key=qdrant_api_key,collection_name=collection_name,force_recreate=False,distance_func="Dot")

    time_taken =time.time() - start

    source = Path(file_path).name

    return 'Source: ' + source + '\n\n' + 'Time Taken To Store in VectorDB: ' + str(f"{time_taken:.2f}") + 'sec'
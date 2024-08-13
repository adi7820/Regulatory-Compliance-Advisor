from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from cleantext import clean
import dotenv
import os
import time
import warnings

warnings.filterwarnings("ignore")
dotenv.load_dotenv()

qdrant_api_key = os.getenv('QDRANT_API_KEY')
hf_token = os.getenv('HF_TOKEN')


def update_qdrant_vectorestore_with_url(collection_name, inp):
    urls = []
    urls.append(inp)

    loader = UnstructuredURLLoader(urls=urls, show_progress_bar=True)
    loaded_url = loader.load()

    raw_text =loaded_url[0].page_content

    clean_url_text = clean(text=raw_text,
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
        "source": loaded_url[0].metadata['source']
    }

    url_doc = Document(page_content=clean_url_text, metadata=metadata)
    url_document =[url_doc]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    url_texts = text_splitter.split_documents(url_document)

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
    qdrant = Qdrant.from_documents(url_texts,embeddings_model,url=url,prefer_grpc=False,api_key=qdrant_api_key,collection_name=collection_name,force_recreate=False,distance_func="Dot")
    time_taken =time.time()-start
    source = loaded_url[0].metadata['source']

    return 'Source: ' + source + '\n\n' + 'Time Taken to Store in VectorDB: ' + str(f"{time_taken:.2f}") + 'sec'
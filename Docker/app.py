import gradio as gr
import os
import dotenv
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from source_wikipedia import update_vectorstore_with_query
from source_pdf import update_vectorstore_with_pdf
from pathlib import Path
from source_url import update_vectorestore_with_url
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.prompts.prompt import PromptTemplate
from qdrant_source_wikipedia import update_qudrant_vectorstore_with_query
from qdrant_source_pdf import update_qdrant_vectorstore_with_pdf
from qdrant_source_url import update_qdrant_vectorestore_with_url
import warnings

warnings.filterwarnings("ignore")
dotenv.load_dotenv()


PROMPT_TEMPLATE = """[INST]You are a friendly virtual assistant and maintain a conversational, polite, patient, friendly, and gender-neutral tone throughout the conversation.

Your task is to understand the QUESTION, review the Content provided in the DOCUMENT, generate a concise and accurate answer based on the Content, and provide references used in answering the question in the format "Source: [URL]".
Do not depend on outside knowledge or fabricate responses.
DOCUMENT: ```{context}```

Please follow these steps:

1. Provide a short, clear, and concise answer.
    * If detailed instructions or steps are required, present them in an ordered list or bullet points.
2. If the answer to the question is not available in the provided DOCUMENT, ONLY respond that you couldn't find any information related to the QUESTION. Do not include references or citations in this case.
3. Citation:
    * Start the citation section with "Here are the Sources used to generate this response." and list the references in markdown link format as "Source: [URL]".
    * Use bullets to list each reference.
    * ONLY use the URL extracted from the DOCUMENT. DO NOT fabricate or use any external links.
    * Avoid over-citation; include only references that were directly used in generating the response.
    * If no references are available, omit the citation section entirely.
    * The Citation section should include one or more references. Do not list the same source multiple times. ALWAYS append the citation section at the end of your response.
    * Example format:
      Here are the Sources used to generate this response:
      * Source: [URL]
[/INST]
[INST]
QUESTION: {question}
FINAL ANSWER:[/INST]
"""

# Set OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
qdrant_api_key = os.getenv('QDRANT_API_KEY')
hf_token = os.getenv('HF_TOKEN')

# Initialize Pinecone with API key
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

# Initialize embeddings
embeddings = OpenAIEmbeddings()
def initialize_embeddings(model_name, model_kwargs, encode_kwargs):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

# Function to create PineconeVectorStore based on index_name
def create_vectorstore(index_name):
    index_name = index_name.lower()  # Convert index name to lowercase
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return vectorstore

# Initialize Qdrant client and vector store
def initialize_qdrant_client(url, api_key, collection_name, embeddings_model):
    client = QdrantClient(url=url, api_key=api_key)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings_model,
        distance="Dot"
    )
    return vector_store

def build_context(chunks):
    context = ""
    for chunk in chunks:
        context = context + "\n  Content: " + chunk.page_content + "| Source: " + chunk.metadata.get("source")
    return context


def generate_answer(llm, vectorstore, prompt_template, question):
    retrieved_chunks = vectorstore.similarity_search(question, k=20)
    context = build_context(retrieved_chunks)
    args = {"context": context, "question": question}
    prompt = prompt_template.format(**args)
    ans = llm.invoke(prompt)
    return ans.content

# Function to run the QA system
def run_qa_system(index_name, query):
    vectorstore = create_vectorstore(index_name)
    
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name='gpt-4o-mini',
        temperature=0.0
    )
    
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vectorstore.as_retriever(search_kwargs={'k': 20})
    # )
    

    prompt_template = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    answer = generate_answer(llm, vectorstore, prompt_template, query)
    # qa.run(query)
    return answer

def run_qudrant_qa_system(collection_name, query):
    if collection_name == "Finance":
        collection_name="finance_collection"
    elif collection_name == "Healthcare":
        collection_name = "Healthcare_collection"
    else:
        collection_name = "DataProtection_collection"

    model_name = 'dunzhang/stella_en_400M_v5'
    model_kwargs = {'device': 'cuda', 'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings_model = initialize_embeddings(model_name, model_kwargs, encode_kwargs)

    vector_store = initialize_qdrant_client(
        url="https://a1fbbfce-ca47-437d-b308-e79724650692.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key=qdrant_api_key,
        collection_name=collection_name,
        embeddings_model=embeddings_model
    )

    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name='gpt-4o-mini',
        temperature=0.0
    )

    prompt_template = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    answer = generate_answer(llm, vector_store, prompt_template, query)

    return answer

# Define Gradio interface
def gradio_interface(index_name, query, vectordb):
    if index_name is None:
        res = gr.Warning("Please Select Domain ⛔️!", duration=5)
    elif vectordb is None:
        res = gr.Warning("Please Select VectorDB ⛔️!", duration=5)
    elif query == "":
        res = gr.Warning("Please Enter Your Query ⛔️!", duration=5)
    elif vectordb == "Qdrant":
        res = run_qudrant_qa_system(index_name, query)
    else:
        res = run_qa_system(index_name, query)
    return res

# Define Wikipedia interface with domain radio buttons
def wikipedia_interface(vectordb, domain, query):
    if domain is None:
        res  = gr.Warning("Please Select Domain ⛔️!", duration=5)
    elif query == "":
        res  = gr.Warning("Please Enter Title ⛔️!", duration=5)
    elif vectordb is None:
        res = gr.Warning("Please Select VectorDB ⛔️!", duration=5)
    elif vectordb == "Qdrant":
        if domain == "Finance":
            collection_name="finance_collection"
        elif domain == "Healthcare":
            collection_name = "Healthcare_collection"
        else:
            collection_name = "DataProtection_collection"
        res = update_qudrant_vectorstore_with_query(query, collection_name)
        gr.Info(f"ℹ️ New Data has been added in Qdrant of {domain} domain", duration=5)
    else:
        domain = domain.lower()
        res = update_vectorstore_with_query(query, domain)
        gr.Info(f"ℹ️ New Data has been added in Pinecone of {domain} domain", duration=5)
        #res = f"Hello {domain}, you selected {query} domain."

    return res

def upload_file(vectordb, domain, filepath):
    if domain is None:
        res  = gr.Warning("Please Select Domain ⛔️!", duration=5)
    elif filepath is None:
        res  = gr.Warning("Please Upload a File ⛔️!", duration=5)
    elif vectordb is None:
        res  = gr.Warning("Please Select VectorDB ⛔️!", duration=5)
    elif vectordb == "Qdrant":
        if domain == "Finance":
            collection_name="finance_collection"
        elif domain == "Healthcare":
            collection_name = "Healthcare_collection"
        else:
            collection_name = "DataProtection_collection"
        res = update_qdrant_vectorstore_with_pdf(filepath, collection_name)
        gr.Info(f"ℹ️ New Data has been added in Qdrant of {domain} domain", duration=5)
    else:
        domain = domain.lower()
        # name = Path(filepath).name
        print(filepath)
        res = update_vectorstore_with_pdf(domain, filepath)
        gr.Info(f"ℹ️ New Data has been added in Pinecone of {domain} domain", duration=5)
    return res

def url_interface(vectordb, domain, url):
    if domain is None:
        res  = gr.Warning("Please Select Domain ⛔️!", duration=5)
    elif url == "":
        res  = gr.Warning("Please Enter a Valid URL ⛔️!", duration=5)
    elif vectordb is None:
        res = gr.Warning("Please Select VectorDB ⛔️!", duration=5)
    elif vectordb == "Qdrant":
        if domain == "Finance":
            collection_name="finance_collection"
        elif domain == "Healthcare":
            collection_name = "Healthcare_collection"
        else:
            collection_name = "DataProtection_collection"
        res = update_qdrant_vectorestore_with_url(collection_name, url)
        gr.Info(f"ℹ️ New Data has been added in Qdrant of {domain} domain", duration=5)
    else:
        domain = domain.lower()
        res = update_vectorestore_with_url(domain, url)
        gr.Info(f"ℹ️ New Data has been added in Pinecone of {domain} domain", duration=5)
    return res

# Import Gradio theme
theme = gr.themes.Soft(
    neutral_hue="slate",
).set(
    body_background_fill='*neutral_50',
    body_background_fill_dark='*neutral_500',
    background_fill_primary_dark='*neutral_50',
    background_fill_secondary_dark='*neutral_50',
    block_background_fill='*primary_50',
    block_background_fill_dark='*primary_100'
)


# wikipedia_interface = gr.Interface(lambda name: "Hello " + name, "text", "text")
wikipedia_interface = gr.Interface(
    fn=wikipedia_interface,
    inputs=[
        gr.Radio(
            choices=["Pinecone", "Qdrant"],
            label="Select VectorDB"),
        gr.Radio(
            choices=["Finance", "Healthcare", "Dataprivacy"],
            label="Select Domain")
        ,
        gr.Textbox(label="Title", placeholder="Enter your wikipedia source title...")
    ],
    outputs=[gr.Textbox(label="Meta Data")]
)

# upload_button = upload_button.upload(upload_file, upload_button)
# pdf_interface = gr.Interface(lambda name: "Hi " + name, "text", "text")
pdf_interface = gr.Interface(
    fn=upload_file,
    inputs=[gr.Radio(
            choices=["Pinecone", "Qdrant"],
            label="Select VectorDB"),
        gr.Radio(
            choices=["Finance", "Healthcare", "Dataprivacy"],
            label="Select Domain")
        ,
        gr.File(label="Upload PDF File", file_types=["pdf"], file_count="single")],
    outputs=gr.Textbox(label="Meta Data")
)

dataprivacy_interface = gr.Interface(
    fn=url_interface,
    inputs=[gr.Radio(
            choices=["Pinecone", "Qdrant"],
            label="Select VectorDB"),
        gr.Radio(
            choices=["Finance", "Healthcare", "Dataprivacy"],
            label="Select Domain"),
            gr.Textbox(label="URL", placeholder="Enter your source URL...")
        ],
    outputs=gr.Textbox(label="Meta Data")
)

# Create Gradio app with theme
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# Regulatory Compliance Advisor")
    
    with gr.Row():
        index_name = gr.Radio(
            choices=["Finance", "Healthcare", "Dataprivacy"],
            label="Select Domain"
        )
        vectordb_name = gr.Radio(
            choices=["Pinecone", "Qdrant"],
            label="Select VectorDB",
            
        )
    
    query = gr.Textbox(
        label="Enter your query",
        placeholder="Enter your query here..."
    )
    
    result = gr.Textbox(label="Result")

    submit_button = gr.Button("Submit")

    submit_button.click(
        fn=gradio_interface,
        inputs=[index_name, query, vectordb_name],
        outputs=result
    )

    # Add parameter viewer with tabbed interface inside the accordion
    with gr.Accordion("Add New Information in RAG Application", open=False):       
        gr.TabbedInterface(
            [wikipedia_interface, pdf_interface, dataprivacy_interface],
            ["Wikipedia", "PDF", "Other URL"]
        )

demo.launch()

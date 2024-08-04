import gradio as gr
import os
import dotenv
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from source_wikipedia import update_vectorstore_with_query
from source_pdf import update_vectorstore_with_pdf
from pathlib import Path

dotenv.load_dotenv()

# Set OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone with API key
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Function to create PineconeVectorStore based on index_name
def create_vectorstore(index_name):
    index_name = index_name.lower()  # Convert index name to lowercase
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return vectorstore

# Function to run the QA system
def run_qa_system(index_name, query):
    vectorstore = create_vectorstore(index_name)
    
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name='gpt-4o-mini',
        temperature=0.0
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa.run(query)

# Define Gradio interface
def gradio_interface(index_name, query):
    return run_qa_system(index_name, query)

# Define Wikipedia interface with domain radio buttons
def wikipedia_interface(domain, query):
    if domain is None:
        res  = gr.Warning("Please Select Domain ⛔️!", duration=5)
    elif query == "":
        res  = gr.Warning("Please Enter Title ⛔️!", duration=5)
    else:
        domain = domain.lower()
        res = update_vectorstore_with_query(query, domain)
        gr.Info(f"New Data has been added in {domain} domain ℹ️", duration=5)
        #res = f"Hello {domain}, you selected {query} domain."

    return res

def upload_file(domain, filepath):
    if domain is None:
        res  = gr.Warning("Please Select Domain ⛔️!", duration=5)
    elif filepath is None:
        res  = gr.Warning("Please Upload a File ⛔️!", duration=5)
    else:
        domain = domain.lower()
        # name = Path(filepath).name
        print(filepath)
        res = update_vectorstore_with_pdf(domain, filepath)
        gr.Info(f"New Data has been added in {domain} domain ℹ️", duration=5)
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
            choices=["Finance", "Healthcare", "Dataprivacy"],
            label="Select Domain")
        ,
        gr.File(label="Upload PDF File", file_types=["pdf"], file_count="single")],
    outputs=gr.Textbox(label="Meta Data")
)
bye_interface = gr.Interface(lambda name: "Bye " + name, "text", "text")

# Create Gradio app with theme
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# Regulatory Compliance Advisor")
    
    index_name = gr.Radio(
        choices=["Finance", "Healthcare", "Dataprivacy"],
        label="Select Index Name"
    )
    
    query = gr.Textbox(
        label="Enter your query",
        placeholder="Enter your query here..."
    )
    
    result = gr.Textbox(label="Result")

    submit_button = gr.Button("Submit")

    submit_button.click(
        fn=gradio_interface,
        inputs=[index_name, query],
        outputs=result
    )

    # Add parameter viewer with tabbed interface inside the accordion
    with gr.Accordion("Add New Information in RAG Application", open=False):       
        gr.TabbedInterface(
            [wikipedia_interface, pdf_interface, bye_interface],
            ["Wikipedia", "PDF", "Other URL"]
        )

demo.launch()

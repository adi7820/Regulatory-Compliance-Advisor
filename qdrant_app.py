from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
import dotenv
import os
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
import warnings
warnings.filterwarnings("ignore")
dotenv.load_dotenv()

PROMPT_TEMPLATE = """[INST]You are a friendly virtual assistant and maintain a conversational, polite, patient, friendly and gender neutral tone throughout the conversation.

Your task is to understand the QUESTION, read the Content list from the DOCUMENT, generate an answer based on the Content, and provide references used in answering the question in the format "Source:".
Do not depend on outside knowledge or fabricate responses.
DOCUMENT: ```{context}```

Your response should follow these steps:

1. The answer should be short and concise, clear.
    * If detailed instructions are required, present them in an ordered list or bullet points.
2. If the answer to the question is not available in the provided DOCUMENT, ONLY respond that you couldn't find any information related to the QUESTION, and do not show references and citations.
3. Citation
    * ALWAYS start the citation section with "Here are the Sources to generate response." and follow with references in markdown link format (Source:) to support the answer.
    * Use Bullets to display the reference (Source: ).
    * You MUST ONLY use the Source extracted from the DOCUMENT as the reference link. DO NOT fabricate or use any link outside the DOCUMENT as reference.
    * Avoid over-citation. Only include references that were directly used in generating the response.
    * If no reference Source can be provided, remove the entire citation section.
    * The Citation section can include one or more references. DO NOT include same Source as multiple references. ALWAYS append the citation section at the end of your response.
    * You MUST follow the below format as an example for this citation section:
      Here are the Sources used to generate this response:
      * Source:
[/INST]
[INST]
QUESTION: {question}
FINAL ANSWER:[/INST]"""

prompt_template = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

qdrant_api_key = os.getenv('QDRANT_API_KEY')
hf_token = os.getenv('HF_TOKEN')

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

collection_name = "finance_collection"

query = "Anti money laundering of EU?"
query_vector = embeddings_model.embed_query(query)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings_model,
    distance="Dot"
)

# completion llm
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name='gpt-4o-mini',
    temperature=0.0
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)
# print(qa.run(query))

def build_context(chunks):
    context = ""
    for chunk in chunks:
        context = context + "\n  Content: " + chunk.page_content + "| Source: " + chunk.metadata.get("source")
    return context

# retrieved_chunks = vector_store.similarity_search(query)
# context = build_context(retrieved_chunks)
# print(context)


def generate_answer(llm, vectorstore, prompt_template, question):
    retrieved_chunks = vectorstore.similarity_search(question, k=20)
    context = build_context(retrieved_chunks)
    print(context)
    args = {"context":context, "question":question}
    prompt = prompt_template.format(**args)
    ans = llm.invoke(prompt)
    return ans.content

print(generate_answer(llm, vector_store, prompt_template, query))
import os
import time
import google.cloud.logging

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import VertexAI
from langchain.chains import RetrievalQA
from langchain.schema import AIMessage, HumanMessage

from pydantic import BaseModel
from typing import Dict, List

import gradio as gr

import vertexai

# Your Google Cloud Project ID
PROJECT_ID = os.environ.get('GCP_PROJECT', None)
print(PROJECT_ID)
# Your Google Cloud Project Region
REGION = 'us-central1'

ENV = os.environ.get('ENV', None)
print(ENV)

if ENV == 'cloud':
    client = google.cloud.logging.Client(project=PROJECT_ID)
    client.setup_logging()

    log_name = "bull-analyst-app-log"
    logger = client.logger(log_name)

elif ENV == 'docker':
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/app/application_default_credentials.json'


def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    print("Waiting")
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            print(".", end="")
            time.sleep(sleep_time)


class CustomVertexAIEmbeddings(VertexAIEmbeddings):
    requests_per_minute: int
    num_instances_per_batch: int

    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch:],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]


qa_chain = None


def create_qa_chain(file):
    # Load document
    loader = PyPDFLoader(file.name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print(docs[-1])

    # Embedding
    EMBEDDING_QPM = 100
    EMBEDDING_NUM_BATCH = 5
    embeddings = CustomVertexAIEmbeddings(
        requests_per_minute=EMBEDDING_QPM,
        num_instances_per_batch=EMBEDDING_NUM_BATCH,
    )

    #  Vetor store
    db = Chroma.from_documents(docs, embeddings)

    # Expose index to the retriever
    retriever = db.as_retriever(search_type="similarity",
                                search_kwargs={"k": 2})

    # LLM model
    llm = VertexAI(
        model_name="text-bison@001",
        max_output_tokens=256,
        temperature=0.1,
        top_p=0.8,
        top_k=40,
        verbose=True,
    )

    global qa_chain

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='refine',
        retriever=retriever,
        return_source_documents=True
    )

    chat.submit_btn.visible = True

    return f"Ready to answer, using {file.name.split('/')[-1]}"


vertexai.init(project=PROJECT_ID, location=REGION)


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    response = qa_chain({"query": message})

    return response['result']


# Gradio UI
with gr.Blocks(theme=gr.themes.Default(text_size="lg")) as demo:

    gr.Image(value=os.path.join(os.path.dirname(
        __file__), "banner.png"), interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1) Upload your 10K pdf OR Choose examples below")
            file = gr.Interface(create_qa_chain, "file",
                                "text")
            gr.Markdown("## 10K Examples")
            gr.Examples(
                [os.path.join(os.path.dirname(__file__),
                              "20230203_alphabet_10K.pdf")],
                inputs=file.input_components,
                outputs=None
            )
        with gr.Column(scale=4):
            chat = gr.ChatInterface(predict)


demo.launch(server_name="0.0.0.0", server_port=8080)  # For Cloud Run

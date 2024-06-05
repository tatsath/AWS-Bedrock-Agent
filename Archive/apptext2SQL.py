from typing import Type
import boto3
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import streamlit as st

## We will be suing Titan Embeddings Model To generate Embedding
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.chat_models.bedrock import BedrockChat

## Data Ingestion
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector Embedding And Vector Store
from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_experimental.graph_transformers import (
    LLMGraphTransformer,
)

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_core.documents import Document
import yfinance as yf

st_callback = StreamlitCallbackHandler(st.container())

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-image-v1", client=bedrock
)


## Data ingestion
def loader():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    docs = [doc for doc in documents if doc.page_content.strip()]
    return docs


def data_ingestion(documents):
    # - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    docs = text_splitter.split_documents(documents)
    return docs


## Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")


def get_titan_llm():
    ##create the Anthropic Model
    llm = BedrockChat(
        model_id="amazon.titan-text-express-v1",
        client=bedrock,
        model_kwargs={"maxTokenCount": 4096},
    )
    return llm


def get_mistral_llm():
    ##create the Anthropic Model
    llm = Bedrock(
        model_id="mistral.mistral-7b-instruct-v0:2",
        client=bedrock,
    )
    llm.model_kwargs = {
        "temperature": 0.3,
        "max_tokens": 1000,
    }
    return llm


def get_llama_llm():
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock)
    llm.model_kwargs = {"max_gen_len": 2048}
    return llm


def get_llm_transformer(llm):
    transformer = LLMGraphTransformer(llm=llm)
    return transformer


def graph_documents(llm, docs):
    transformer = get_llm_transformer(llm)
    # text = """
    # Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
    # She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
    # Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
    # She was, in 1906, the first woman to become a professor at the University of Paris.
    # """
    # documents = [Document(page_content=text)]
    # graph_docs = transformer.convert_to_graph_documents(documents=documents)
    graph_docs = transformer.convert_to_graph_documents(documents=docs)
    print(f"Nodes; {graph_docs[0].nodes}")
    print(f"relationships; {graph_docs[0].relationships}")
    return graph_docs


# llm = get_llama_llm()
# docs = loader()
# graph_documents(llm, docs)


def get_current_stock_price(ticker):
    ticker_data = yf.Ticker(ticker)
    recent = ticker_data.history(period="1d")
    return recent.iloc[0]["Close"]


class CurrentStockPriceInput(BaseModel):
    """Inputs for get_current_stock_price"""

    ticker: str = Field(description="Ticker symbol of the stock")


class CurrentStockPriceTool(BaseTool):
    name = "get_current_stock_price"
    description = "Useful when you want to get current stock price"
    args_schema: Type[BaseModel] = CurrentStockPriceInput  # type: ignore

    def _run(self, ticker):
        return get_current_stock_price(ticker)

    def _arun(self, ticker):
        raise NotImplementedError("func get_current_stock_price did not support async.")


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.

{agent_scratchpad}

<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    answer = qa({"query": query})
    return answer["result"]


def configure_retriever():
    bedrock_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-image-v1", client=bedrock
    )
    vectorstore = FAISS.load_local(
        "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore


latest_stock_price = CurrentStockPriceTool(
    name="get_current_stock_price",
    description="Get the latest stock price for a given ticker symbol.",
)

# print("init retriever tool")


def search_docs(query):
    """Searches the document store for relevant information."""
    vectorstore = configure_retriever()
    print(f"query: {query}")
    # if query["value"]:
    #     results = vectorstore.similarity_search(query["value"])
    # else:
    results = vectorstore.similarity_search(query["query"])
    return {"docs": results}


retriever_tool = Tool(
    name="search_docs",
    func=search_docs,
    description="Search the document store for relevant information.",
)

# print("init tools")
tools = [retriever_tool, latest_stock_price]
print("init openai functions")
# llm = get_mistral_llm()
llm = get_mistral_llm()
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=ConversationBufferMemory(),
    verbose=True,
)
# print("init agent_executor")
# agent_executor = AgentExecutor(
#     agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
# )
# print("questions: Hello")
# agent_executor.invoke({"input": "Hello?", "agent_scratchpad": ""})
# print("questions: What is machine learning")
# agent_executor.invoke({"input": "What is machine learning?", "agent_scratchpad": ""})
# print("questions: AAPL stock price")
# agent_executor.invoke(
#     {"input": "What is the latest stock price of AAPL?", "agent_scratchpad": ""}
# )


def main():
    # st.set_page_config("Chat PDF")

    st.header("Agent")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion(loader())
                get_vector_store(docs)
                st.success("Done")

    if st.button("Run Agent"):
        with st.spinner("Processing..."):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent.invoke(
                {"input": user_question}, {"callbacks": [st_callback]}
            )
            st.write(response["output"])
            st.success("Done")


if __name__ == "__main__":
    main()

from dotenv import load_dotenv
import os

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

from FlagEmbedding import FlagReranker

from src.retrieval_prompt import contextualize_q_system_prompt

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

index_name = "medicalchatbot"

# reuse helper function
embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

reranker = FlagReranker('BAAI/bge-reranker-base')

def retrieve_and_rerank(query):

    docs = retriever.invoke(query)

    scores = reranker.compute_score(
        [[query, doc.page_content] for doc in docs]
    )

    ranked_docs = [
        doc for _, doc in sorted(
            zip(scores, docs),
            reverse=True
        )
    ][:3]

    context_text = ""   

    for doc in ranked_docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        context_text += f"\nSource: {source}\n{doc.page_content}\n"

    return context_text

llm=ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://models.inference.ai.azure.com",
    api_key=OPENAI_API_KEY,
    temperature=0.7,
    streaming=True
)



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ]
)

rewrite_chain = contextualize_q_prompt | llm | StrOutputParser()

rag_chain = (
    {
        "context": lambda x: retrieve_and_rerank(
            rewrite_chain.invoke(
                {
                    "input": x["input"],
                    "chat_history": x.get("chat_history", [])
                }
            )
        ),
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# store chat histories
chat_histories = {}

def get_session_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

# chain with memory
rag_chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
)
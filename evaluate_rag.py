from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from datasets import Dataset
from langchain_openai import ChatOpenAI
import os

from src.ragchain import rag_chain
from src.ragchain import retrieve_and_rerank

# Example evaluation questions
questions = [
    "What is diabetes?",
    "What are symptoms of diabetes?",
    "How is diabetes treated?"
]

answers = []
contexts = []

# Generate answers from your RAG system
for q in questions:

    # retrieve context
    context = retrieve_and_rerank(q)

    # generate answer using the RAG chain
    answer = rag_chain.invoke({"input": q})

    answers.append(answer)
    contexts.append([context])

# Ground truth answers (expected answers)
ground_truth = [
    "Diabetes is a metabolic disorder characterized by high blood sugar.",
    "Symptoms include increased thirst, frequent urination, and fatigue.",
    "Treatment includes insulin therapy, medication, and lifestyle changes."
]

dataset = Dataset.from_dict({
    "question": questions,
    "response": answers,
    "contexts": contexts,
    "ground_truth": ground_truth
})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://models.inference.ai.azure.com",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

result = evaluate(
    dataset,
    metrics=[
    Faithfulness(),
    AnswerRelevancy(),
    ContextPrecision()
],
    llm=llm
)

print(result)
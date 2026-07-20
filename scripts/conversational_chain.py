from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "crops"

RELEVANCE_THRESHOLD = 0.4
NO_INFO_MESSAGE = "I don't have information about that in the crop knowledge base."

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about crop
cultivation using ONLY the information given in the context below. The context
may be in Bangla or English.

Rules:
- Answer in the same language the user asked in.
- Be concise and practical — this is for farmers, not an essay.

Context:
{context}
"""

CONTEXTUALIZE_PROMPT = """Given the conversation so far and a new user question,
rewrite the new question as a standalone question that makes sense without the
conversation history. Do NOT answer the question. If the question is already
standalone, return it unchanged.

Only output the rewritten question, nothing else."""

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def build_pipeline():
    embeddings = OllamaEmbeddings(model="bge-m3")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    llm = ChatOllama(model="llama3.2:3b", temperature=0)

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])
    generation_chain = answer_prompt | llm | StrOutputParser()


    return vectorstore, contextualize_chain, generation_chain

def answer(question: str, history: list, vectorstore, contextualize_chain, generation_chain, k: int=4):
    if history:
        standalone_question = contextualize_chain.invoke({"history": history, "question":question})
    else:
        standalone_question = question
    print(f"[standalone question] {standalone_question}")
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(standalone_question, k=k)
    for doc, score in docs_with_scores:
        print(f" {score:.3f} | {doc.metadata['crop_name']} ({doc.metadata['section']})")
    
    top_score = max((s for _,s in docs_with_scores), default=0)
    if top_score < RELEVANCE_THRESHOLD:
        return NO_INFO_MESSAGE
    docs = [d for d,_ in docs_with_scores]
    print("\n===== CONTEXT =====")
    print(format_docs(docs))
    print("===================\n")
    reply =  generation_chain.invoke({
        "context": format_docs(docs), 
        "history": history,
        "question": question
    })
    return reply

if __name__ == "__main__": 
    vectorstore, contextualize_chain, generation_chain = build_pipeline()
    history = []
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        reply = answer(question, history, vectorstore, contextualize_chain, generation_chain)
        print(f"\nBot: {reply}\n")

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=reply))

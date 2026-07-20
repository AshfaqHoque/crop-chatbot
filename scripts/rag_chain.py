from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def build_pipeline():
    embeddings = OllamaEmbeddings(model="bge-m3")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    generation_chain = prompt | llm | StrOutputParser()
    return vectorstore, generation_chain

def answer(question: str, vectorstore, generation_chain, k: int=4):
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    for doc, score in docs_with_scores:
        print(f" {score:.3f} | {doc.metadata['crop_name']} ({doc.metadata['section']})")
    
    top_score = max((s for _,s in docs_with_scores), default=0)
    if top_score < RELEVANCE_THRESHOLD:
        return NO_INFO_MESSAGE
    docs = [d for d,_ in docs_with_scores]
    return generation_chain.invoke({"context": format_docs(docs), "question": question})

if __name__ == "__main__": 
    vectorstore, generation_chain = build_pipeline()
    while True:
        question = input("You: ").strip()
        reply = answer(question, vectorstore, generation_chain)

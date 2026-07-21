from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.retrievers import BM25Retriever

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "crops"

RELEVANCE_THRESHOLD = 0.3
NO_INFO_MESSAGE = "I don't have information about that in the crop knowledge base."

SYSTEM_PROMPT = """You are a strict closed-book assistant. The CONTEXT below is
your ONLY source of truth. You have no other knowledge about crops.
 
CONTEXT:
{context}
 
ABSOLUTE RULES:
1. Every number, date, quantity, pest/disease name, or method in your answer
   must come directly from the CONTEXT above (translated if needed) — never
   from what you already know about farming in general.
2. If the user asks about something the CONTEXT does not cover (a specific
   number, a growing month, a pest, anything), say plainly: "The knowledge
   base doesn't specify that" for that part. Do NOT estimate, calculate, or
   supply a plausible-sounding value to fill the gap.
3. Before writing your final answer, check every number and named detail you
   are about to include against the CONTEXT. If you cannot point to where in
   the CONTEXT it came from, delete it.
4. The CONTEXT may contain information about MULTIPLE different crops (each
   chunk is labeled with its crop name in brackets, e.g. "[Aman Rice / আমন
   ধান]"). Never combine or transfer a fact from one crop to another, even if
   their names look or sound similar (e.g. Aman rice vs Aam/mango are
   different crops). Only use facts labeled with the crop the user asked about.
5. Answer in the same language the user asked in.
6. Be concise. Summarize the key actionable points from the CONTEXT in your
   own words — do not copy long passages verbatim. Cover the most important
   points across ALL relevant retrieved sections (not just one), but keep the
   total answer under ~120 words, using short bullets.
7. Answer only the exact information requested. A context that merely mentions
   an entity name or provides its control method does not necessarily define
   what the entity is. If the requested definition, cause, symptom, or other
   specific information is missing, clearly say that the knowledge base does
   not specify it. Do not convert related information into an unsupported
   direct answer.
"""

CONTEXTUALIZE_PROMPT = """Given the conversation so far and a new user question,
rewrite the new question as a standalone question that makes sense without the
conversation history. Do NOT answer the question. If the question is already
standalone, return it unchanged.

Only output the rewritten question, nothing else."""

def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('crop_name', 'Unknown')}] "
        f"({doc.metadata.get('section', 'Unknown')})\n"
        f"{doc.page_content}"
        for doc in docs
    )

def build_pipeline():
    embeddings = OllamaEmbeddings(model="bge-m3")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    stored = vectorstore.get(include=["documents", "metadatas"])
    bm25_retriever = BM25Retriever(stored["documents"], metadatas=stored["metadatas"])
    bm25_retriever.k = 4
    
    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    generation_chain = answer_prompt | llm | StrOutputParser()

    return vectorstore, bm25_retriever, contextualize_chain, generation_chain

def answer(question: str, history: list, vectorstore, bm25_retriever, contextualize_chain, generation_chain, k: int=4):
    if history:
        standalone_question = contextualize_chain.invoke({"history": history, "question":question})
    else:
        standalone_question = question
    print(f"[standalone question] {standalone_question}")
    # ? vector search
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(standalone_question, k=k)

    # top_score = max((s for _,s in docs_with_scores), default=0)
    # if top_score < RELEVANCE_THRESHOLD:
    #     return NO_INFO_MESSAGE
    
    vector_docs = [d for d,score in docs_with_scores]
    bm25_docs = bm25_retriever.invoke(standalone_question)

    combined_docs = []

    for doc in vector_docs + bm25_docs:
        if doc.page_content not in [d.page_content for d in combined_docs]:
            combined_docs.append(doc)

    docs = combined_docs[:4]

    print("\n===== VECTOR =====")
    for doc, score in docs_with_scores:
        print(f"{score:.3f} | {doc.metadata['crop_name']} ({doc.metadata['section']})")

    print("\n===== BM25 =====")
    for i, doc in enumerate(bm25_docs, 1):
        print(f"{i}. {doc.metadata['crop_name']} ({doc.metadata['section']})")
        
    reply =  generation_chain.invoke({
        "context": format_docs(docs), 
        "question": standalone_question
    })
    return reply

if __name__ == "__main__": 
    vectorstore, bm25_retriever, contextualize_chain, generation_chain = build_pipeline()
    history = []
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        reply = answer(question, history, vectorstore, bm25_retriever, contextualize_chain, generation_chain)
        print(f"\nBot: {reply}\n")

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=reply))





from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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
    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        # MessagesPlaceholder("history"),
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
    docs = [d for d,score in docs_with_scores if score >= RELEVANCE_THRESHOLD]
    print("\n===== CONTEXT =====")
    print(format_docs(docs))
    print("===================\n")
    reply =  generation_chain.invoke({
        "context": format_docs(docs), 
        # "history": history,
        "question": standalone_question
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

















# from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_community.retrievers import BM25Retriever

# import json
# import re


# PERSIST_DIR = "chroma_db"
# COLLECTION_NAME = "crops"

# DENSE_CANDIDATE_K = 10
# BM25_CANDIDATE_K = 10
# FINAL_CONTEXT_K = 4

# DENSE_WEIGHT = 0.6
# BM25_WEIGHT = 0.4
# RRF_CONSTANT = 60

# RELEVANCE_THRESHOLD = 0.3

# NO_INFO_MESSAGE = "I don't have information about that in the crop knowledge base."


# SYSTEM_PROMPT = """You are a strict closed-book assistant. The CONTEXT below is
# your ONLY source of truth. You have no other knowledge about crops.

# CONTEXT:
# {context}

# ABSOLUTE RULES:
# 1. Every number, date, quantity, pest/disease name, or method in your answer
#    must come directly from the CONTEXT above (translated if needed) — never
#    from what you already know about farming in general.
# 2. If the user asks about something the CONTEXT does not cover (a specific
#    number, a growing month, a pest, anything), say plainly: "The knowledge
#    base doesn't specify that" for that part. Do NOT estimate, calculate, or
#    supply a plausible-sounding value to fill the gap.
# 3. Before writing your final answer, check every number and named detail you
#    are about to include against the CONTEXT. If you cannot point to where in
#    the CONTEXT it came from, delete it.
# 4. The CONTEXT may contain information about MULTIPLE different crops (each
#    chunk is labeled with its crop name in brackets, e.g. "[Aman Rice / আমন
#    ধান]"). Never combine or transfer a fact from one crop to another, even if
#    their names look or sound similar (e.g. Aman rice vs Aam/mango are
#    different crops). Only use facts labeled with the crop the user asked about.
# 5. Answer in the same language the user asked in.
# 6. Be concise. Summarize the key actionable points from the CONTEXT in your
#    own words — do not copy long passages verbatim. Cover the most important
#    points across ALL relevant retrieved sections (not just one), but keep the
#    total answer under ~120 words, using short bullets.
# 7. Answer only the exact information requested. A context that merely mentions
#    an entity name or provides its control method does not necessarily define
#    what the entity is. If the requested definition, cause, symptom, or other
#    specific information is missing, clearly say that the knowledge base does
#    not specify it. Do not convert related information into an unsupported
#    direct answer.
# """


# CONTEXTUALIZE_PROMPT = """Given the conversation so far and a new user question,
# rewrite the new question as a standalone question that makes sense without the
# conversation history. Do NOT answer the question. If the question is already
# standalone, return it unchanged.

# Only output the rewritten question, nothing else."""


# def format_docs(docs) -> str:
#     formatted_docs = []

#     for doc in docs:
#         crop_name = doc.metadata.get("crop_name", "Unknown crop")
#         section = doc.metadata.get("section", "Unknown section")

#         formatted_docs.append(f"[{crop_name}] ({section})\n{doc.page_content}")

#     return "\n\n".join(formatted_docs)


# def get_document_key(doc) -> str:
#     chunk_id = doc.metadata.get("chunk_id")

#     if chunk_id:
#         return str(chunk_id)

#     return doc.page_content


# def reciprocal_rank_fusion(
#     vector_docs,
#     bm25_docs,
#     final_k=FINAL_CONTEXT_K,
# ):
#     document_scores = {}
#     document_map = {}

#     ranked_lists = [
#         (vector_docs, DENSE_WEIGHT),
#         (bm25_docs, BM25_WEIGHT),
#     ]

#     for docs, weight in ranked_lists:
#         seen_documents = set()

#         for rank, doc in enumerate(docs, start=1):
#             document_key = get_document_key(doc)

#             if document_key in seen_documents:
#                 continue

#             seen_documents.add(document_key)
#             document_map[document_key] = doc

#             rrf_score = weight / (RRF_CONSTANT + rank)

#             document_scores[document_key] = (
#                 document_scores.get(document_key, 0) + rrf_score
#             )

#     ranked_document_keys = sorted(
#         document_scores,
#         key=document_scores.get,
#         reverse=True,
#     )

#     final_docs = [
#         document_map[document_key] for document_key in ranked_document_keys[:final_k]
#     ]

#     return final_docs


# def check_bm25_hit(
#     question: str,
#     bm25_docs,
# ) -> bool:
#     if not bm25_docs:
#         return False

#     query_words = {
#         word for word in re.findall(r"\w+", question.casefold()) if len(word) > 2
#     }

#     if not query_words:
#         return False

#     for doc in bm25_docs:
#         document_words = set(
#             re.findall(
#                 r"\w+",
#                 doc.page_content.casefold(),
#             )
#         )

#         if query_words.intersection(document_words):
#             return True

#     return False


# def build_pipeline():
#     embeddings = OllamaEmbeddings(model="bge-m3")

#     vectorstore = Chroma(
#         collection_name=COLLECTION_NAME,
#         embedding_function=embeddings,
#         persist_directory=PERSIST_DIR,
#     )

#     with open(
#         "data/documents.json",
#         encoding="utf-8",
#     ) as file:
#         raw_docs = json.load(file)

#     texts = [document["text"] for document in raw_docs]

#     metadatas = [document["metadata"] for document in raw_docs]

#     bm25_retriever = BM25Retriever.from_texts(
#         texts,
#         metadatas=metadatas,
#     )

#     bm25_retriever.k = BM25_CANDIDATE_K

#     llm = ChatOllama(
#         model="qwen2.5:7b-instruct",
#         temperature=0,
#         num_predict=256
#     )

#     contextualize_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 CONTEXTUALIZE_PROMPT,
#             ),
#             MessagesPlaceholder("history"),
#             (
#                 "human",
#                 "{question}",
#             ),
#         ]
#     )

#     contextualize_chain = contextualize_prompt | llm | StrOutputParser()

#     answer_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 SYSTEM_PROMPT,
#             ),
#             (
#                 "human",
#                 "{question}",
#             ),
#         ]
#     )

#     generation_chain = answer_prompt | llm | StrOutputParser()

#     return (
#         vectorstore,
#         bm25_retriever,
#         contextualize_chain,
#         generation_chain,
#     )


# def answer(
#     question: str,
#     history: list,
#     vectorstore,
#     bm25_retriever,
#     contextualize_chain,
#     generation_chain,
# ):
#     if history:
#         standalone_question = contextualize_chain.invoke(
#             {
#                 "history": history,
#                 "question": question,
#             }
#         )
#     else:
#         standalone_question = question

#     standalone_question = standalone_question.strip()

#     print(f"[standalone question] {standalone_question}")

#     # Dense semantic search
#     docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
#         standalone_question,
#         k=DENSE_CANDIDATE_K,
#     )

#     print("\n===== VECTOR RESULTS =====")

#     for rank, (doc, score) in enumerate(
#         docs_with_scores,
#         start=1,
#     ):
#         crop_name = doc.metadata.get(
#             "crop_name",
#             "Unknown",
#         )

#         section = doc.metadata.get(
#             "section",
#             "Unknown",
#         )

#         print(f"{rank}. {score:.3f} | {crop_name} ({section})")

#     vector_top_score = max(
#         (score for _, score in docs_with_scores),
#         default=0,
#     )

#     vector_hit = vector_top_score >= RELEVANCE_THRESHOLD

#     vector_docs = [doc for doc, _ in docs_with_scores]

#     print(f"[vector top score] {vector_top_score:.3f}")

#     print(f"[vector hit] {vector_hit}")

#     # BM25 keyword search
#     bm25_docs = bm25_retriever.invoke(standalone_question)

#     print("\n===== BM25 RESULTS =====")

#     for rank, doc in enumerate(
#         bm25_docs,
#         start=1,
#     ):
#         crop_name = doc.metadata.get(
#             "crop_name",
#             "Unknown",
#         )

#         section = doc.metadata.get(
#             "section",
#             "Unknown",
#         )

#         print(f"{rank}. {crop_name} ({section})")

#     bm25_hit = check_bm25_hit(
#         question=standalone_question,
#         bm25_docs=bm25_docs,
#     )

#     print(f"[bm25 hit] {bm25_hit}")

#     if not vector_hit and not bm25_hit:
#         return NO_INFO_MESSAGE

#     valid_vector_docs = [
#         doc
#         for doc, score in docs_with_scores
#         if score >= RELEVANCE_THRESHOLD
#     ]

#     valid_bm25_docs = bm25_docs if bm25_hit else []

#     docs = reciprocal_rank_fusion(
#         vector_docs=valid_vector_docs,
#         bm25_docs=valid_bm25_docs,
#         final_k=FINAL_CONTEXT_K,
#     )

#     if not docs:
#         return NO_INFO_MESSAGE

#     print("\n===== FINAL FUSED CONTEXT =====")

#     for rank, doc in enumerate(
#         docs,
#         start=1,
#     ):
#         crop_name = doc.metadata.get(
#             "crop_name",
#             "Unknown",
#         )

#         section = doc.metadata.get(
#             "section",
#             "Unknown",
#         )

#         print(f"{rank}. {crop_name} ({section})")

#     print("===============================\n")
#     context_text = format_docs(docs)

#     print("\n===== ACTUAL CONTEXT SENT TO LLM =====")
#     print(context_text)
#     print("======================================\n")
    
#     reply = generation_chain.invoke(
#         {
#             "context": format_docs(docs),
#             "question": standalone_question,
#         }
#     )

#     return reply


# if __name__ == "__main__":
#     (
#         vectorstore,
#         bm25_retriever,
#         contextualize_chain,
#         generation_chain,
#     ) = build_pipeline()

#     history = []

#     while True:
#         question = input("You: ").strip()

#         if question.lower() in {
#             "exit",
#             "quit",
#         }:
#             break

#         if not question:
#             continue

#         reply = answer(
#             question=question,
#             history=history,
#             vectorstore=vectorstore,
#             bm25_retriever=bm25_retriever,
#             contextualize_chain=contextualize_chain,
#             generation_chain=generation_chain,
#         )

#         print(f"\nBot: {reply}\n")

#         history.append(HumanMessage(content=question))

#         history.append(AIMessage(content=reply))

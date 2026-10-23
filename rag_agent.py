from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


# 1. Create embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 2. Load ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# 3. Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# 4. Create LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# 5. Function to answer questions
def ask_question(question):

    # Search for relevant chunks
    documents = retriever.invoke(question)

    # If no relevant information is found
    if not documents:
        return "I could not find this information in the provided documents."

    # Combine retrieved document chunks
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Prompt the LLM
    prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question ONLY using the information
provided in the document context below.

If the answer is not present in the context, say:

"I could not find this information in the provided documents."

Do not use outside knowledge.
Do not make up information.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    # Generate answer
    response = llm.invoke(prompt)

    return response.content


# 6. Run the chatbot
while True:

    question = input("\nAsk your question: ")

    # Exit chatbot
    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Get answer
    answer = ask_question(question)

    # Display answer
    print("\nAnswer:")
    print(answer)
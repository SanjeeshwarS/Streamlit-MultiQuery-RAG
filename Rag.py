from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
import ollama
import os



doc_path = "./data/Galaxies.pdf"
model = "llama3.2"
embedding_model = "nomic-embed-text"


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        print("File Successfully Loaded......\n")
        return data
    else:
        raise FileNotFoundError(f"PDF file not found at path: {doc_path}")


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300, languages = ["eng"])
    chunks = text_splitter.split_documents(documents)
    print("Documents split into chunks......\n")
    return chunks


def create_vector_db(chunks):
    """Create a vector database from document chunks."""
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=embedding_model),
        collection_name="galaxy-facts",
    )
    print("Vector database created......")
    return vector_db


def create_retriever(vector_db, llm):

    """Create a multi-query retriever."""

    """The Query Prompt is to make the llm create 5 different versions of the same question and then it is used to search in the vector_db"""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever = vector_db.as_retriever(), 
        llm = llm, 
        prompt = QUERY_PROMPT
    )
    print("Retriever created.....\n")
    return retriever


def create_chain(retriever, llm):
    """Create the chain"""

    # RAG prompt 
    template = """Answer the question based ONLY on the following context:
        {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Chain created successfully.")
    return chain


def main():
    # Load and process the PDF document
    data = ingest_pdf(doc_path)

    # Split the documents into chunks
    chunks = split_documents(data)

    # Create the vector database
    vector_db = create_vector_db(chunks)

    # Initialize the  model
    llm = ChatOllama(model=model)

    # Create the retriever
    retriever = create_retriever(vector_db, llm)

    #Rag Chain is created
    chain = create_chain(retriever, llm)

    #Question
    question = "Tell me about the Andromeda Galaxy?"

    #Printing the Response
    res = chain.invoke(question)
    print("Response:")
    print(res)


if __name__ == "__main__":
    main()
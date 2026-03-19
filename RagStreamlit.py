from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
import streamlit as st
import ollama
import os



doc_path = "./data/Galaxies.pdf"
model = "llama3.2"
embedding_model = "nomic-embed-text"
embedding = OllamaEmbeddings(model = "nomic-embed-text")
persist_directory = "./my_chroma_db"



def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path = doc_path)
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

@st.cache_resource
def get_llm():
    llm = ChatOllama(model = model)
    return llm


@st.cache_resource
def load_vectordb(active_path):

    if os.path.exists(persist_directory):
        vector_db = Chroma(
            embedding_function = embedding,
            collection_name = "galaxy-facts",
            persist_directory = persist_directory,
        )
        print("Loaded existing documents successfully......")
        return vector_db
    else:
        data = ingest_pdf(active_path)
        if data is None:
            return None
        
        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents = chunks,
            embedding = embedding,
            collection_name = "galaxy-facts",
            persist_directory = persist_directory,
        )
        print("Vector Database Successfully Created......")

        return vector_db


def create_retriever(vector_db, llm):

    """Create a multi-query retriever."""

    #The Query Prompt is to make the llm create 5 different versions of the same question and then it is used to search in the vector_db
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
    st.title("Sanjii's Rag System")
    
    st.subheader("Advanced Multi-Query PDF Assistant")
    st.markdown("""Developed by **Sanjii**, this platform leverages **State-of-the-Art Retrieval-Augmented Generation (RAG)** to provide precise, context-driven insights from your PDF documents.
    
    # Key Technical Features:
        -Multi-Query Synthesis:** Automatically generates 5 diverse search perspectives to overcome similarity search limitations.
        -Local Privacy:** Powered by **Ollama (Llama 3.2)** and **ChromaDB** for secure, local-first data processing.
        -Intelligent Vector Retrieval:** Optimized chunking using Recursive Character Splitting for better context retention.
    """)
    st.divider()

    with st.sidebar:
        st.header("Upload Center")
        upload_file = st.file_uploader("Upload Your PDF", type="pdf")
        st.divider()
        st.header("System Status")
        st.success("✅ Ollama: Running (Llama 3.2)")
        
    if upload_file:

        if not os.path.exists("./data"):
            os.makedirs("./data")

        active_path = os.path.join("./data", upload_file.name)
        with open(active_path,  "wb") as f:
            f.write(upload_file.getbuffer())
    else:
        active_path = doc_path

    vector_db = None
    if active_path:
        with st.sidebar:
            with st.status("🔄 Processing Vector DB...", expanded=True) as status:
                vector_db = load_vectordb(active_path)
                if vector_db:
                    num_chunks = len(vector_db.get()['ids'])
                    status.update(label=f"📂 {num_chunks} Chunks Loaded!", state="complete", expanded=False)
                    st.info(f"Database contains {num_chunks} chunks.")
                else:
                    status.update(label="⚠️ Loading Failed", state="error")
    

    user_input = st.text_input("Enter Your Question:", )

    if user_input:
        if vector_db is None:
            st.error("Please Wait For The Database To Load.")
        else:
            with st.status(" Analysing your question....", expanded = True) as run_status:
                try:
                
                    llm = get_llm()

                    run_status.write(" Generating search perspectives (Multi-Query)...")
                    vector_db = load_vectordb(active_path)
            
                    #Create Retriver
                    run_status.write(" Retrieving relevant documents...")
                    retriever = create_retriever(vector_db, llm)

                    #Create Chain
                    run_status.write("✍️ Synthesizing final answer...")
                    chain = create_chain(retriever, llm)

                    #Get Response
                    response = chain.invoke(input=user_input)
                    run_status.update(label="✅ Response Ready!", state="complete", expanded=False)

                    st.markdown("**Assistant**")
                    st.write(response)

                    with st.expander(" View Source Documents "):
                        docs = retriever.invoke(user_input)
                        for i, doc in enumerate(docs):
                            st.info(f"Chunk {i+0}:\n\n{doc.page_content}")
                except Exception as e:
                    st.error(f"An Error Occured: {str(e)}")

    else:
        st.info("Please Enter A Question To Get Started......")



if __name__ == "__main__":
    main()
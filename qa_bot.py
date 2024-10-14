from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Load PDF
loader = DirectoryLoader("Data/", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

text_chunks = text_splitter.split_documents(documents)

# Create Embeddings and Vector Database
# Create the embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device':"cpu"}
)

# Create vectorstore
vector_store = FAISS.from_documents(text_chunks,embeddings)

# Setting the retriever
retriever = vector_store.as_retriever(search_kwargs={"k":2})

# Function for prompt
def chatbot_prompt():
    # Define the prompt template
    general_system_template = r"""
    You are an assistant designed to answer questions based on healthcare from users.

    **Layer 1: Contextual Understanding**
    Please read the following user reviews carefully and provide precise answers to the questions based on the source document context provided. Do not answer any questions outside this context.

    **Layer 2: Response Guidelines**
    If the answer is not explicitly found in the source document context, kindly state: "I'm sorry, I don't have that information." Please do not fabricate any answers or discuss unrelated topics.
    ----
    {context}
    ----
    """
    
    general_user_template = "Question:```{question}```"
    
    messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    
    # Create the PromptTemplate
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    
    return qa_prompt

def qa_chain():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={'max_new_tokens':128,'temperature':0.01}
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key= "answer" # Specify that we want to store 'answer' in memory
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,chain_type='stuff',
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': chatbot_prompt()}
    )
    
    return chain


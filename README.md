# Healthcare ChatBot 🩺

This repository contains a **Healthcare ChatBot** built using **LangChain** and **Streamlit**. The chatbot is designed to help users ask healthcare-related questions and receive answers based on the document content provided (e.g., healthcare PDFs).

## Project Overview

This project uses LangChain to integrate a language model, which retrieves and generates responses based on healthcare documents. It leverages **FAISS** for efficient document chunk retrieval and **CTransformers** to handle the Llama 2 language model for generating accurate responses. Users can interact with the chatbot through a friendly Streamlit interface.

### Key Features:
- **Conversational Memory**: The chatbot maintains a memory of previous conversations to ensure a smooth dialogue with the user.
- **Document-based Q&A**: It retrieves and provides answers based on healthcare-related PDFs.
- **LangChain and Streamlit Integration**: Seamlessly integrates LangChain for conversation management and Streamlit for the front-end.
- **FAISS Vector Store**: Efficient document retrieval system to find relevant chunks of text from loaded documents.
- **Embeddings**: Uses **HuggingFaceEmbeddings** for embedding the documents and performing similarity searches.

---

### Preview of the ChatBot Interface

![ChatBot UI](https://drive.google.com/uc?export=view&id=1uLD8OVDNZzsEB-80hY34Ndaf6Uj8hx2U)

---

### System Architecture

![System Architecture](https://drive.google.com/uc?export=view&id=1C6azciutSCauX8jjSGyj0FV6IHd7GN-0)

The architecture is structured as follows:
1. **Ingestion**: The PDFs are ingested and split into smaller chunks to facilitate retrieval.
2. **Embeddings and Vector Store**: The chunks are embedded using Hugging Face's embeddings and stored in a FAISS vector store.
3. **Query and Response Generation**: When the user submits a query, the system retrieves the most relevant chunks and uses the Llama 2 model to generate an appropriate response.

---

## How to Run the Project

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/healthcare-chatbot.git
cd healthcare-chatbot
```

### 2. Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app:
```bash
streamlit run app.py
```

You can now interact with the chatbot through the Streamlit interface. The chatbot will respond to queries based on the documents in the `Data/` folder

## Technologies Used

- **LangChain**: For creating and managing conversational logic.
- **Streamlit**: To build the web interface.
- **FAISS**: For fast document retrieval.
- **CTransformers**: To run the Llama 2 model for generating responses.
- **Hugging Face Embeddings**: For creating document embeddings to facilitate similarity-based search.

---

## Future Improvements:

- Expand the chatbot to handle other file types (e.g., Word documents, CSVs).
- Implement more sophisticated conversation flows and responses.
- Support more advanced healthcare domains and conditions.

---

### Source:
Source video for this project: [YouTube Video](https://youtu.be/XNmFIkViEBU?si=FxJ_8BmBxrDldyVC)








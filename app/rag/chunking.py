from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag.loader import load_data

# data = load_data("./data")

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       
        chunk_overlap=50,  
        length_function=len,
        add_start_index=True, 
    )
    chunks = splitter.split_documents(docs)
    # print(f"Tổng số chunks sau khi cắt: {len(chunks)}")
    return chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


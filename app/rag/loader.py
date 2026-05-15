from langchain_community.document_loaders import CSVLoader, DirectoryLoader
import os

def load_data(directory_path = "./data"):
    abs_path = os.path.abspath(directory_path)
    # print(f"--- Đang quét thư mục: {abs_path} ---")

    loader = DirectoryLoader(
        path=abs_path,
        glob="**/*.csv",
        loader_cls=CSVLoader,
        loader_kwargs={
            "encoding": "utf-8", # Thử utf-8 trước
            "csv_args": {
                "delimiter": "," # Đảm bảo đúng dấu phẩy
            }
        }
    )
    
    docs = loader.load()
    # print(f"Tổng số Documents load được từ LangChain: {len(docs)}")
    return docs
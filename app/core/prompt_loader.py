def load_prompt(file_path="prompts/rag_prompt.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
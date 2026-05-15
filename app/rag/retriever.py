def retrieve(db, query, k=10):
    return db.similarity_search(query, k=k)
"""Agent truy xuat tai lieu lien quan tu vector store."""

from langchain_core.vectorstores import VectorStore

from rag_engine.agents.history import build_retrieval_query
from rag_engine.agents.state import AgentState
from rag_engine.rag.reranker import get_candidate_k
from rag_engine.rag.retriever import retrieve


def make_retrieval_agent(db: VectorStore, default_top_k: int):
    """Tao retrieval agent gan voi mot vector store va top_k mac dinh."""

    def retrieval_agent(state: AgentState) -> AgentState:
        """Tim tai lieu lien quan, ghep context va thu thap danh sach nguon."""
        top_k = int(state.get("top_k") or default_top_k)
        retrieval_query = build_retrieval_query(
            state["query"],
            state.get("history", []),
        )
        docs = retrieve(db, retrieval_query, k=get_candidate_k(top_k))
        context = "\n\n".join(doc.page_content for doc in docs)
        sources = sorted(
            {
                str(doc.metadata.get("source"))
                for doc in docs
                if doc.metadata.get("source")
            }
        )

        return {
            **state,
            "retrieved_docs": docs,
            "context": context,
            "sources": sources,
            "retrieval_query": retrieval_query,
        }

    return retrieval_agent

"""Agent sap xep lai tai lieu truoc khi sinh cau tra loi."""

from rag_engine.agents.state import AgentState
from rag_engine.rag.reranker import rerank_documents


def make_reranker_agent(default_top_k: int):
    """Tao reranker agent dua tren query va danh sach tai lieu da retrieve."""

    def reranker_agent(state: AgentState) -> AgentState:
        """Rerank docs, cap nhat context va danh sach source cuoi cung."""
        docs = state.get("retrieved_docs") or []
        top_k = int(state.get("top_k") or default_top_k)
        reranked_docs = rerank_documents(state["query"], docs, top_k=top_k)
        context = "\n\n".join(doc.page_content for doc in reranked_docs)
        sources = sorted(
            {
                str(doc.metadata.get("source"))
                for doc in reranked_docs
                if doc.metadata.get("source")
            }
        )

        return {
            **state,
            "retrieved_docs": reranked_docs,
            "context": context,
            "sources": sources,
        }

    return reranker_agent

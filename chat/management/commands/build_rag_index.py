"""Django command to build or rebuild the vector index for RAG."""

from django.core.management.base import BaseCommand

from rag_engine.core.config import settings
from rag_engine.ingestion.build_index import build_index
from rag_engine.rag.vector_store import count_vectors


class Command(BaseCommand):
    """CLI command that builds TXT product data into the vector index."""

    help = "Build or rebuild the vector index from local product TXT data."

    def add_arguments(self, parser):
        """Declare CLI arguments for the data and index directories."""
        parser.add_argument("--data-dir", default=str(settings.data_dir))
        parser.add_argument("--index-dir", default=str(settings.faiss_index_dir))

    def handle(self, *args, **options):
        """Run the index build and print the resulting vector count."""
        db = build_index(
            data_dir=options["data_dir"],
            index_dir=options["index_dir"],
        )
        backend = settings.vector_backend
        self.stdout.write(
            self.style.SUCCESS(
                f"Built {backend} index with {count_vectors(db)} vectors."
            )
        )

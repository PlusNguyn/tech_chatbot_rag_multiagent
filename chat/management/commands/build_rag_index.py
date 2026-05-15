"""Django command dùng để tạo hoặc tạo lại FAISS index cho RAG."""

from django.core.management.base import BaseCommand

from rag_engine.core.config import settings
from rag_engine.ingestion.build_index import build_index


class Command(BaseCommand):
    """Command CLI build dữ liệu CSV thành FAISS vector index."""

    help = "Build or rebuild the FAISS index from local product CSV data."

    def add_arguments(self, parser):
        """Khai báo các tham số dòng lệnh cho đường dẫn dữ liệu và index."""
        parser.add_argument("--data-dir", default=str(settings.data_dir))
        parser.add_argument("--index-dir", default=str(settings.faiss_index_dir))

    def handle(self, *args, **options):
        """Chạy quá trình build index và in số lượng vector đã tạo."""
        db = build_index(
            data_dir=options["data_dir"],
            index_dir=options["index_dir"],
        )
        self.stdout.write(
            self.style.SUCCESS(f"Built FAISS index with {db.index.ntotal} vectors.")
        )

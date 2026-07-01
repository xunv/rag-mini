import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import indexer


class IndexerMetadataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.db_path = self.tmp_path / "rag.db"
        self.docs_path = self.tmp_path / "docs"
        self.docs_path.mkdir()
        self.original_db_path = indexer.DB_PATH
        indexer.DB_PATH = str(self.db_path)

    def tearDown(self):
        indexer.DB_PATH = self.original_db_path
        self.tmp.cleanup()

    def test_relative_source_path_uses_path_under_data_dir(self):
        nested = self.docs_path / "sub" / "same.txt"
        nested.parent.mkdir()
        nested.write_text("hello", encoding="utf-8")

        result = indexer._relative_source_path(str(nested), str(self.docs_path))

        self.assertEqual(result, "sub/same.txt")

    def test_init_db_creates_document_metadata_columns(self):
        conn = indexer.init_db(rebuild=True)

        chunk_cols = [row["name"] for row in conn.execute("PRAGMA table_info(chunks)")]
        doc_cols = [row["name"] for row in conn.execute("PRAGMA table_info(documents)")]

        self.assertIn("document_id", chunk_cols)
        self.assertIn("source_path", chunk_cols)
        self.assertIn("file_hash", doc_cols)
        self.assertIn("chunk_count", doc_cols)
        conn.close()

    def test_migrates_legacy_chunks_into_documents(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE chunks (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                text_content TEXT NOT NULL,
                chapter TEXT DEFAULT '',
                chapter_title TEXT DEFAULT '',
                source_file TEXT NOT NULL,
                file_hash TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO chunks (
                text_content, chapter, chapter_title, source_file, file_hash
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("旧文本。", "", "", "legacy.txt", "hash-a"),
        )
        conn.commit()
        conn.close()

        conn = indexer.init_db(rebuild=False)
        hashes = indexer.get_indexed_file_hashes(conn)
        row = conn.execute("SELECT document_id, source_path FROM chunks").fetchone()

        self.assertEqual(hashes, {"legacy.txt": "hash-a"})
        self.assertEqual(row["source_path"], "legacy.txt")
        self.assertIsNotNone(row["document_id"])
        conn.close()

    def test_cleanup_deleted_files_removes_documents_and_chunks(self):
        conn = indexer.init_db(rebuild=True)
        file_path = self.docs_path / "missing.txt"
        file_path.write_text("第一句内容足够长。第二句内容足够长。", encoding="utf-8")

        with (
            patch.object(indexer, "parse_file", return_value=[
                {"text": "待删除文本内容足够长。", "chapter": "", "chapter_title": ""}
            ]),
            patch.object(indexer, "get_embeddings_batch", return_value=[[0.1] * indexer.VECTOR_DIMS]),
        ):
            indexer.index_file(conn, str(file_path), "hash-a", str(self.docs_path))

        deleted = indexer.cleanup_deleted_files(conn, set())

        self.assertEqual(deleted, 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 0)
        conn.close()


if __name__ == "__main__":
    unittest.main()

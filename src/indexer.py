"""索引构建脚本 — 扫描 docs/ 目录，增量入库（SQLite + sqlite-vec + FTS5）"""

import os
import hashlib
import logging
import argparse
import sqlite3
import struct
import time
from pathlib import Path

# 兼容直接运行 (python src/indexer.py) 和模块运行 (python -m src.indexer)
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

import sqlite_vec
import jieba

from .config import (
    DB_PATH,
    DATA_DIR,
    VECTOR_DIMS,
)
from .models import get_embeddings_batch
from .parsers import SUPPORTED_EXTENSIONS, parse_file

logger = logging.getLogger(__name__)


def _serialize_vector(vec: list[float]) -> bytes:
    """将 float 列表序列化为 sqlite-vec 需要的 bytes 格式"""
    return struct.pack(f"{len(vec)}f", *vec)


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """检查表字段是否存在，用于兼容旧库迁移"""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def _relative_source_path(file_path: str, data_dir: str) -> str:
    """生成稳定的文档相对路径，避免同名文件冲突"""
    path = Path(file_path).resolve()
    base = Path(data_dir).resolve()
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


# ==================== 数据库初始化 ====================


def init_db(rebuild: bool = False) -> sqlite3.Connection:
    """初始化 SQLite 数据库，创建表结构"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    if rebuild and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        logger.info(f"已删除旧数据库: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row

    # 主表：存储文档片段
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            text_content TEXT NOT NULL,
            chapter TEXT DEFAULT '',
            chapter_title TEXT DEFAULT '',
            source_file TEXT NOT NULL,
            source_path TEXT DEFAULT '',
            file_hash TEXT NOT NULL
        )
    """)

    # 兼容旧数据库：补齐新增字段
    if not _column_exists(conn, "chunks", "document_id"):
        conn.execute("ALTER TABLE chunks ADD COLUMN document_id INTEGER")
    if not _column_exists(conn, "chunks", "source_path"):
        conn.execute("ALTER TABLE chunks ADD COLUMN source_path TEXT DEFAULT ''")

    # 文档元数据表：用于增量判断、同名文件区分和删除清理
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT NOT NULL UNIQUE,
            source_file TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            mtime REAL NOT NULL,
            size_bytes INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            indexed_at REAL NOT NULL
        )
    """)

    # 向量表：sqlite-vec 虚拟表
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding float[{VECTOR_DIMS}]
        )
    """)

    # FTS5 全文索引表（存储 jieba 分词后的文本，提升中文 BM25 效果）
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text_content,
            content='chunks',
            content_rowid='rowid',
            tokenize='unicode61'
        )
    """)

    # 不使用触发器同步 FTS5 — 改为在 index_file 中手动写入分词后文本，
    # 因为 SQLite 触发器无法调用 Python jieba 分词

    conn.commit()
    migrate_document_metadata(conn)
    logger.info(f"数据库已就绪: {DB_PATH}")
    return conn


# ==================== 文件扫描 ====================


def scan_files(data_dir: str) -> list[str]:
    """扫描目录下所有支持的文件"""
    files = []
    base = Path(data_dir)
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(str(path))
    logger.info(f"扫描到 {len(files)} 个文件: {[Path(f).name for f in files]}")
    return files


def compute_file_hash(file_path: str) -> str:
    """计算文件的 MD5 哈希，用于增量判断"""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_indexed_file_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    """查询已索引文件的哈希值，返回 {source_path: file_hash}"""
    rows = conn.execute(
        "SELECT source_path, file_hash FROM documents"
    ).fetchall()
    return {row["source_path"]: row["file_hash"] for row in rows}


def migrate_document_metadata(conn: sqlite3.Connection) -> None:
    """将旧库中只有 chunks 的文件信息迁移到 documents 表"""
    rows = conn.execute("""
        SELECT
            COALESCE(NULLIF(source_path, ''), source_file) AS source_path,
            source_file,
            file_hash,
            COUNT(*) AS chunk_count
        FROM chunks
        WHERE document_id IS NULL
        GROUP BY COALESCE(NULLIF(source_path, ''), source_file), source_file, file_hash
    """).fetchall()
    if not rows:
        return

    now = time.time()
    for row in rows:
        conn.execute(
            """
            INSERT INTO documents (
                source_path, source_file, file_hash, mtime, size_bytes, chunk_count, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                source_file=excluded.source_file,
                file_hash=excluded.file_hash,
                chunk_count=excluded.chunk_count,
                indexed_at=excluded.indexed_at
            """,
            (
                row["source_path"],
                row["source_file"],
                row["file_hash"],
                0.0,
                0,
                row["chunk_count"],
                now,
            ),
        )
        doc_id = conn.execute(
            "SELECT id FROM documents WHERE source_path = ?", (row["source_path"],)
        ).fetchone()["id"]
        conn.execute(
            """
            UPDATE chunks
            SET document_id = ?, source_path = ?
            WHERE document_id IS NULL
              AND COALESCE(NULLIF(source_path, ''), source_file) = ?
              AND source_file = ?
              AND file_hash = ?
            """,
            (doc_id, row["source_path"], row["source_path"], row["source_file"], row["file_hash"]),
        )
    conn.commit()
    logger.info(f"已迁移旧文档元数据: {len(rows)} 个文件")


def delete_file_docs(conn: sqlite3.Connection, source_path: str) -> int:
    """删除某个文件的所有文档（含向量和 FTS）"""
    # 获取要删除的 rowid
    rowids = [
        row[0] for row in conn.execute(
            "SELECT rowid FROM chunks WHERE source_path = ?", (source_path,)
        ).fetchall()
    ]
    if not rowids:
        conn.execute("DELETE FROM documents WHERE source_path = ?", (source_path,))
        conn.commit()
        return 0

    # 删除向量表
    for rid in rowids:
        conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rid,))

    # 删除 FTS5 索引（内容同步模式需手动删除）
    # 先获取原始文本用于 FTS5 删除
    for rid in rowids:
        row = conn.execute("SELECT text_content FROM chunks WHERE rowid = ?", (rid,)).fetchone()
        if row:
            segmented = " ".join(jieba.cut(row["text_content"]))
            conn.execute(
                "INSERT INTO chunks_fts(chunks_fts, rowid, text_content) VALUES('delete', ?, ?)",
                (rid, segmented),
            )

    # 删除主表
    conn.execute("DELETE FROM chunks WHERE source_path = ?", (source_path,))
    conn.execute("DELETE FROM documents WHERE source_path = ?", (source_path,))
    conn.commit()

    logger.info(f"已删除文件 '{source_path}' 的 {len(rowids)} 条旧文档")
    return len(rowids)


def cleanup_deleted_files(conn: sqlite3.Connection, current_source_paths: set[str]) -> int:
    """清理已从文档目录删除的文件索引"""
    rows = conn.execute("SELECT source_path FROM documents").fetchall()
    stale_paths = [row["source_path"] for row in rows if row["source_path"] not in current_source_paths]
    deleted_chunks = 0
    for source_path in stale_paths:
        deleted_chunks += delete_file_docs(conn, source_path)
    if stale_paths:
        logger.info(f"已清理不存在的文件索引: {len(stale_paths)} 个文件, {deleted_chunks} 条片段")
    return deleted_chunks


# ==================== 索引写入 ====================


def index_file(conn: sqlite3.Connection, file_path: str, file_hash: str, data_dir: str) -> int:
    """对单个文件进行切片、向量化、写入 SQLite"""
    source_file = os.path.basename(file_path)
    source_path = _relative_source_path(file_path, data_dir)
    stat = os.stat(file_path)
    logger.info(f"正在处理文件: {source_path}")

    chunks = parse_file(file_path)
    if not chunks:
        logger.warning(f"文件 '{source_path}' 无有效内容，跳过")
        delete_file_docs(conn, source_path)
        return 0

    logger.info(f"  切分为 {len(chunks)} 个片段，开始向量化...")
    texts = [c["text"] for c in chunks]
    vectors = get_embeddings_batch(texts, prompt_name="document", batch_size=32)
    logger.info(f"  向量化完成，写入数据库...")

    delete_file_docs(conn, source_path)

    now = time.time()
    cur = conn.execute(
        """
        INSERT INTO documents (
            source_path, source_file, file_hash, mtime, size_bytes, chunk_count, indexed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (source_path, source_file, file_hash, stat.st_mtime, stat.st_size, len(chunks), now),
    )
    document_id = cur.lastrowid

    # 批量写入：先插入主表获取 rowid，再批量插入向量表和 FTS5
    chunk_rows = [
        (
            document_id,
            c["text"],
            c.get("chapter", c.get("section", "")),
            c.get("chapter_title", c.get("section_title", "")),
            source_file,
            source_path,
            file_hash,
        )
        for c in chunks
    ]
    conn.executemany(
        """
        INSERT INTO chunks (
            document_id, text_content, chapter, chapter_title, source_file, source_path, file_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        chunk_rows,
    )

    # 获取刚插入的 rowid 范围
    last_rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    first_rowid = last_rowid - len(chunks) + 1

    # 批量插入向量
    vec_rows = [
        (first_rowid + i, _serialize_vector(vec))
        for i, vec in enumerate(vectors)
    ]
    conn.executemany(
        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
        vec_rows,
    )

    # 批量插入 FTS5（jieba 分词后的文本）
    fts_rows = [
        (first_rowid + i, " ".join(jieba.cut(c["text"])))
        for i, c in enumerate(chunks)
    ]
    conn.executemany(
        "INSERT INTO chunks_fts(rowid, text_content) VALUES (?, ?)",
        fts_rows,
    )

    conn.commit()
    logger.info(f"  写入完成: {len(chunks)} 条")
    return len(chunks)


# ==================== 主流程 ====================


def rebuild_fts(conn: sqlite3.Connection) -> int:
    """仅重建 FTS5 索引（jieba 分词），不重新向量化，速度很快"""
    # 清空旧 FTS5
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
    conn.commit()
    logger.info("已清空旧 FTS5 索引，开始重建...")

    # 分批读取主表文本，jieba 分词后写入 FTS5
    batch_size = 500
    offset = 0
    total = 0
    while True:
        rows = conn.execute(
            "SELECT rowid, text_content FROM chunks ORDER BY rowid LIMIT ? OFFSET ?",
            (batch_size, offset),
        ).fetchall()
        if not rows:
            break
        fts_rows = [(row["rowid"], " ".join(jieba.cut(row["text_content"]))) for row in rows]
        conn.executemany(
            "INSERT INTO chunks_fts(rowid, text_content) VALUES (?, ?)",
            fts_rows,
        )
        total += len(fts_rows)
        offset += batch_size
        logger.info(f"  FTS5 重建进度: {total} 条")

    conn.commit()
    logger.info(f"FTS5 重建完成: {total} 条")
    return total


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="RAG 索引构建工具（支持增量入库）")
    parser.add_argument("--rebuild", action="store_true", help="删除旧数据库并全量重建")
    parser.add_argument("--rebuild-fts", action="store_true", help="仅重建 FTS5 索引（jieba 分词），不重新向量化")
    parser.add_argument("--dir", default=DATA_DIR, help=f"文档目录（默认: {DATA_DIR}）")
    args = parser.parse_args()

    if args.rebuild_fts:
        # 快速路径：只重建 FTS5，跳过向量化
        conn = init_db(rebuild=False)
        rebuild_fts(conn)
        conn.close()
        return

    conn = init_db(rebuild=args.rebuild)

    files = scan_files(args.dir)
    if not files:
        cleanup_deleted_files(conn, set())
        conn.close()
        logger.warning("未找到任何支持的文件")
        return

    indexed_hashes = get_indexed_file_hashes(conn) if not args.rebuild else {}
    current_source_paths = {_relative_source_path(path, args.dir) for path in files}
    if not args.rebuild:
        cleanup_deleted_files(conn, current_source_paths)

    total_indexed = 0
    skipped = 0

    for file_path in files:
        source_path = _relative_source_path(file_path, args.dir)
        file_hash = compute_file_hash(file_path)

        if source_path in indexed_hashes and indexed_hashes[source_path] == file_hash:
            logger.info(f"文件 '{source_path}' 未变更，跳过")
            skipped += 1
            continue

        count = index_file(conn, file_path, file_hash, args.dir)
        total_indexed += count

    conn.close()
    logger.info(f"\n入库完成: 新增/更新 {total_indexed} 条，跳过 {skipped} 个未变更文件")


if __name__ == "__main__":
    main()

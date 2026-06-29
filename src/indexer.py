"""索引构建脚本 — 扫描 docs/ 目录，支持 txt/pdf，增量入库（SQLite + sqlite-vec + FTS5）"""

import os
import re
import hashlib
import logging
import argparse
import sqlite3
import struct

# 兼容直接运行 (python src/indexer.py) 和模块运行 (python -m src.indexer)
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

import fitz  # pymupdf
import sqlite_vec
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    DB_PATH,
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    VECTOR_DIMS,
)
from .embedding import get_embeddings_batch

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def _serialize_vector(vec: list[float]) -> bytes:
    """将 float 列表序列化为 sqlite-vec 需要的 bytes 格式"""
    return struct.pack(f"{len(vec)}f", *vec)


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
            text_content TEXT NOT NULL,
            chapter TEXT DEFAULT '',
            chapter_title TEXT DEFAULT '',
            source_file TEXT NOT NULL,
            file_hash TEXT NOT NULL
        )
    """)

    # 向量表：sqlite-vec 虚拟表
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding float[{VECTOR_DIMS}]
        )
    """)

    # FTS5 全文索引表
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text_content,
            content='chunks',
            content_rowid='rowid'
        )
    """)

    # FTS5 同步触发器
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text_content) VALUES (new.rowid, new.text_content);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text_content) VALUES('delete', old.rowid, old.text_content);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text_content) VALUES('delete', old.rowid, old.text_content);
            INSERT INTO chunks_fts(rowid, text_content) VALUES (new.rowid, new.text_content);
        END;
    """)

    conn.commit()
    logger.info(f"数据库已就绪: {DB_PATH}")
    return conn


# ==================== 文件扫描 ====================


def scan_files(data_dir: str) -> list[str]:
    """扫描目录下所有支持的文件"""
    files = []
    for entry in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(entry)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(os.path.join(data_dir, entry))
    logger.info(f"扫描到 {len(files)} 个文件: {[os.path.basename(f) for f in files]}")
    return files


def compute_file_hash(file_path: str) -> str:
    """计算文件的 MD5 哈希，用于增量判断"""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_indexed_file_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    """查询已索引文件的哈希值，返回 {source_file: file_hash}"""
    rows = conn.execute(
        "SELECT DISTINCT source_file, file_hash FROM chunks"
    ).fetchall()
    return {row["source_file"]: row["file_hash"] for row in rows}


def delete_file_docs(conn: sqlite3.Connection, source_file: str) -> int:
    """删除某个文件的所有文档（含向量和 FTS）"""
    # 获取要删除的 rowid
    rowids = [
        row[0] for row in conn.execute(
            "SELECT rowid FROM chunks WHERE source_file = ?", (source_file,)
        ).fetchall()
    ]
    if not rowids:
        return 0

    # 删除向量表
    for rid in rowids:
        conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rid,))

    # 删除主表（触发器会自动同步 FTS）
    conn.execute("DELETE FROM chunks WHERE source_file = ?", (source_file,))
    conn.commit()

    logger.info(f"已删除文件 '{source_file}' 的 {len(rowids)} 条旧文档")
    return len(rowids)


# ==================== 文档加载 ====================


def extract_chapter_info(line: str) -> tuple[str, str] | None:
    """从行文本中提取章节号和标题"""
    match = re.match(r"第(\d+)章\s+(.+?)(?:\r|\n|$)", line.strip())
    if match:
        return f"第{match.group(1)}章", match.group(2).strip()
    return None


def load_txt(file_path: str) -> list[dict]:
    """读取 txt 文件，按章节分段再切片"""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chapter_blocks: list[tuple[str, str, str]] = []
    current_chapter = ""
    current_title = ""
    current_lines: list[str] = []

    for line in raw_text.split("\n"):
        info = extract_chapter_info(line)
        if info:
            if current_chapter and current_lines:
                chapter_blocks.append((current_chapter, current_title, "\n".join(current_lines)))
            current_chapter, current_title = info
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_chapter and current_lines:
        chapter_blocks.append((current_chapter, current_title, "\n".join(current_lines)))

    if not chapter_blocks:
        chapter_blocks = [("", "", raw_text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=CHUNK_SEPARATORS,
    )

    results = []
    for chapter, title, content in chapter_blocks:
        chunks = splitter.split_text(content)
        for chunk in chunks:
            if not chunk.strip():
                continue
            results.append({"text": chunk, "chapter": chapter, "chapter_title": title})
    return results


def load_pdf(file_path: str) -> list[dict]:
    """读取 PDF 文件，按页提取文本再切片"""
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            full_text += text + "\n"
    doc.close()

    if not full_text.strip():
        logger.warning(f"PDF 文件 '{file_path}' 未提取到文本内容")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=CHUNK_SEPARATORS,
    )
    chunks = splitter.split_text(full_text)
    return [{"text": c, "chapter": "", "chapter_title": ""} for c in chunks if c.strip()]


def load_file(file_path: str) -> list[dict]:
    """根据文件扩展名选择加载器"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return load_txt(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    raise ValueError(f"不支持的文件格式: {ext}")


# ==================== 索引写入 ====================


def index_file(conn: sqlite3.Connection, file_path: str, file_hash: str) -> int:
    """对单个文件进行切片、向量化、写入 SQLite"""
    source_file = os.path.basename(file_path)
    logger.info(f"正在处理文件: {source_file}")

    chunks = load_file(file_path)
    if not chunks:
        logger.warning(f"文件 '{source_file}' 无有效内容，跳过")
        return 0

    logger.info(f"  切分为 {len(chunks)} 个片段，开始向量化...")
    texts = [c["text"] for c in chunks]
    vectors = get_embeddings_batch(texts, batch_size=32)
    logger.info(f"  向量化完成，写入数据库...")

    # 批量写入：先插入主表获取 rowid，再批量插入向量表
    chunk_rows = [
        (c["text"], c["chapter"], c["chapter_title"], source_file, file_hash)
        for c in chunks
    ]
    conn.executemany(
        "INSERT INTO chunks (text_content, chapter, chapter_title, source_file, file_hash) VALUES (?, ?, ?, ?, ?)",
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

    conn.commit()
    logger.info(f"  写入完成: {len(chunks)} 条")
    return len(chunks)


# ==================== 主流程 ====================


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="RAG 索引构建工具（支持增量入库）")
    parser.add_argument("--rebuild", action="store_true", help="删除旧数据库并全量重建")
    parser.add_argument("--dir", default=DATA_DIR, help=f"文档目录（默认: {DATA_DIR}）")
    args = parser.parse_args()

    conn = init_db(rebuild=args.rebuild)

    files = scan_files(args.dir)
    if not files:
        logger.warning("未找到任何支持的文件")
        return

    indexed_hashes = get_indexed_file_hashes(conn) if not args.rebuild else {}

    total_indexed = 0
    skipped = 0

    for file_path in files:
        source_file = os.path.basename(file_path)
        file_hash = compute_file_hash(file_path)

        if source_file in indexed_hashes and indexed_hashes[source_file] == file_hash:
            logger.info(f"文件 '{source_file}' 未变更，跳过")
            skipped += 1
            continue

        if source_file in indexed_hashes:
            delete_file_docs(conn, source_file)

        count = index_file(conn, file_path, file_hash)
        total_indexed += count

    conn.close()
    logger.info(f"\n入库完成: 新增/更新 {total_indexed} 条，跳过 {skipped} 个未变更文件")


if __name__ == "__main__":
    main()
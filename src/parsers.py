"""文档解析层。

将不同文件格式先通过 Docling 解析为统一 Markdown，再做通用结构识别和切片。
indexer 只负责增量判断、向量化和写库，不关心具体文件格式。
"""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .splitter import split_text

logger = logging.getLogger(__name__)

# Docling 支持的常见文档格式。音视频解析通常依赖额外运行环境，暂不默认扫描。
DOCLING_EXTENSIONS = {
    ".txt",
    ".text",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".odt",
    ".ods",
    ".odp",
    ".epub",
    ".html",
    ".htm",
    ".xhtml",
    ".md",
    ".markdown",
    ".qmd",
    ".rmd",
    ".adoc",
    ".asciidoc",
    ".tex",
    ".latex",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".eml",
    ".msg",
    ".dclg",
}

SUPPORTED_EXTENSIONS = DOCLING_EXTENSIONS

_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_CN_NUMBERED_HEADING_RE = re.compile(
    r"^\s*((?:第[0-9零〇一二三四五六七八九十百千万]+[编章节条款回])|"
    r"(?:卷[0-9零〇一二三四五六七八九十百千万]+))[\s:：、.．]*(.*?)\s*$"
)


@dataclass
class SectionBlock:
    text: str
    section: str = ""
    section_title: str = ""
    section_level: int = 0


def _strip_markdown_inline(text: str) -> str:
    """清理标题里的少量 Markdown 标记，便于作为元数据展示"""
    text = text.strip()
    text = re.sub(r"^[>*\-\s]+", "", text)
    text = re.sub(r"[*_`]+", "", text)
    return text.strip()


def _parse_section_heading(line: str) -> tuple[str, str, int] | None:
    """识别 Markdown 标题和常见中文编号标题"""
    stripped = line.strip()
    if not stripped:
        return None

    level = 1
    heading_text = stripped
    markdown_match = _MARKDOWN_HEADING_RE.match(stripped)
    if markdown_match:
        level = len(markdown_match.group(1))
        heading_text = markdown_match.group(2).strip()

    heading_text = _strip_markdown_inline(heading_text)
    numbered_match = _CN_NUMBERED_HEADING_RE.match(heading_text)
    if numbered_match:
        section = numbered_match.group(1).strip()
        title = numbered_match.group(2).strip()
        return section, title, level

    if markdown_match:
        return heading_text, "", level

    return None


def _split_sections(text: str) -> list[SectionBlock]:
    """从统一文本/Markdown 中识别结构块"""
    lines = text.splitlines()
    blocks: list[SectionBlock] = []
    current_section = ""
    current_title = ""
    current_level = 0
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            blocks.append(
                SectionBlock(
                    text=content,
                    section=current_section,
                    section_title=current_title,
                    section_level=current_level,
                )
            )
        current_lines = []

    for line in lines:
        info = _parse_section_heading(line)
        if info:
            flush()
            current_section, current_title, current_level = info
            current_lines = [line]
        else:
            current_lines.append(line)

    flush()

    if not blocks and text.strip():
        blocks.append(SectionBlock(text=text.strip()))
    return blocks


def _split_text_with_sections(text: str) -> list[dict]:
    """统一结构识别后切片，并兼容旧 chapter 字段"""
    results: list[dict] = []
    for block in _split_sections(text):
        chunks = split_text(block.text)
        for chunk in chunks:
            if chunk.strip():
                results.append(
                    {
                        "text": chunk,
                        "section": block.section,
                        "section_title": block.section_title,
                        "section_level": block.section_level,
                        # 兼容当前数据库和评测字段，后续可迁移为 section_*。
                        "chapter": block.section,
                        "chapter_title": block.section_title,
                    }
                )
    return results


@lru_cache(maxsize=1)
def _get_docling_converter():
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError(
            "Docling 未安装，无法解析该文档格式。请先安装依赖：uv sync 或 pip install docling"
        ) from exc
    return DocumentConverter()


def parse_with_docling(file_path: str) -> list[dict]:
    """使用 Docling 解析文档为 Markdown，再进入通用结构识别和切片流程"""
    converter = _get_docling_converter()
    result = converter.convert(Path(file_path))
    markdown = result.document.export_to_markdown()

    if not markdown.strip():
        logger.warning(f"Docling 未从文件 '{file_path}' 提取到文本内容")
        return []

    return _split_text_with_sections(markdown)


def parse_file(file_path: str) -> list[dict]:
    """根据文件扩展名校验格式，并统一使用 Docling 解析"""
    ext = Path(file_path).suffix.lower()
    if ext in DOCLING_EXTENSIONS:
        return parse_with_docling(file_path)
    raise ValueError(f"不支持的文件格式: {ext}")

"""文本切片模块 — 结构/规则切片。

按段落和句子累积到目标长度，并保留少量 overlap。切片阶段不调用
embedding，索引构建只需要对最终 chunk 做一轮向量化。
"""

import re
from typing import List

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)

# 中文断句标点
_SENT_DELIM = re.compile(r'([。！？；…\.\!?;])([\"\'」』]{0,2})')
# 中文/英文逗号（用于二次拆分过长句子）
_COMMA_DELIM = re.compile(r'([，,])([\"\'」』]{0,2})')


def _normalize_text(text: str) -> str:
    """压缩多余空白，保留段落边界用于规则切片"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_sentences(text: str) -> List[str]:
    """第一步：按标点初切成句子"""
    text = _SENT_DELIM.sub(r'\1\2\n', text)
    sents = [s.strip() for s in text.split('\n') if s.strip()]
    return sents


def _split_balanced_units(text: str) -> List[str]:
    """按段落和句末标点拆成规则切片单元，不调用 embedding"""
    text = _normalize_text(text)
    if not text:
        return []

    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sents = _split_into_sentences(paragraph)
        if sents:
            units.extend(sents)
        else:
            units.append(paragraph)
    return units


def _split_long_text(text: str, chunk_size: int) -> List[str]:
    """长文本兜底拆分，优先按逗号，否则按固定长度"""
    parts = [
        p.strip()
        for p in _COMMA_DELIM.sub(r"\1\2\n", text).split("\n")
        if p.strip()
    ]
    chunks: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            chunks.append(part)
            continue
        for i in range(0, len(part), chunk_size):
            chunk = part[i : i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
    return chunks


def _overlap_tail(text: str, overlap: int) -> str:
    """取上一个 chunk 尾部作为下一块的重叠上下文"""
    if overlap <= 0 or not text:
        return ""
    return text[-overlap:]


def _split_text_balanced(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """结构/规则切片：按句子合并到目标大小，并保留少量 overlap"""
    units = _split_balanced_units(text)
    chunks: list[str] = []
    current = ""

    for unit in units:
        if len(unit) > chunk_size:
            if len(current) >= 10:
                chunks.append(current)
                current = _overlap_tail(current, chunk_overlap)
            for piece in _split_long_text(unit, chunk_size):
                if len(piece) >= 10:
                    if current and len(current) + len(piece) <= chunk_size:
                        current += piece
                    else:
                        if len(current) >= 10:
                            chunks.append(current)
                        current = _overlap_tail(piece, chunk_overlap) if len(piece) >= chunk_size else piece
            continue

        if not current:
            current = unit
            continue

        if len(current) + len(unit) <= chunk_size:
            current += unit
            continue

        if len(current) >= 10:
            chunks.append(current)
        overlap_text = _overlap_tail(current, chunk_overlap)
        current = overlap_text + unit if overlap_text else unit

    if len(current) >= 10:
        chunks.append(current)

    return chunks


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """按结构和标点切片，不在切片阶段调用 embedding。"""
    return _split_text_balanced(text, chunk_size, chunk_overlap)

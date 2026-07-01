# 核心参数调优指南

本文档对应当前实现：sqlite-vec + SQLite FTS5 + 规则切片 + RRF 融合 + Reranker 精排。

---

## 模型后端

### EMBED_BACKEND / RERANKER_BACKEND / LLM_BACKEND

- **默认值**: `ollama`
- **可选值**: `ollama` / `local`
- **说明**: embedding 与 reranker 可在 Ollama API 和本地 SentenceTransformer/CrossEncoder 之间切换。LLM 当前只实现了 Ollama 后端。

| 参数 | 影响 |
| ---- | ---- |
| `ollama` | 部署简单，统一走 Ollama 服务；需要确认 Ollama 模型名和向量维度匹配 |
| `local` | 直接加载本地模型，适合离线和更精细的模型控制；启动时会占用更多本机内存/显存 |

---

## 切片参数

### CHUNK_SIZE / CHUNK_OVERLAP

- **默认值**: `700` / `100`
- **说明**: 按段落/句子累积到 `CHUNK_SIZE` 附近，相邻 chunk 保留 `CHUNK_OVERLAP` 字符上下文。切片阶段不调用 embedding，入库时只对最终 chunk 做一轮向量化。

| 现象 | 调整方向 |
| ---- | -------- |
| chunk 太短、索引慢 | 增大 `CHUNK_SIZE` |
| 答案跨边界丢上下文 | 增大 `CHUNK_OVERLAP` |
| 检索结果重复多 | 减小 `CHUNK_OVERLAP` |

---

## 已弃用：semantic 语义切片

semantic 方案曾用于质量实验：先对句子或缓冲句做 embedding，通过相邻向量距离寻找语义断点，再对最终 chunk 做 embedding 入库。它的问题是重建索引时存在两轮向量化，且会显著增加 chunk 数量。

在当前数据集实验中：

| 方案 | chunk 数 | MRR | 结论 |
| ---- | -------- | --- | ---- |
| semantic 语义切片 | 约 `15162` | 约 `0.518` | 重建慢，质量未提升 |
| 规则切片 `700/100` | 约 `1727` | 约 `0.608` | 更快，质量更好 |

规则切片评测结果还包括：来源命中率 `80.0%`，答案片段命中率 `70.0%`。因此 `SPLITTER_MODE`、`SENTENCE_SIZE`、`SEMANTIC_BUFFER_SIZE`、`SEMANTIC_THRESHOLD_TYPE`、`SEMANTIC_THRESHOLD_AMOUNT` 已从配置中移除。

---

## 检索参数

### SEARCH_TOP_K

- **默认值**: `15`
- **说明**: `search()` 对外返回的最大片段数。当前默认 reranker 只保留 `RERANKER_TOP_K=5`，因此最终用于生成的片段数通常由 reranker 决定。

### RECALL_TOP_K

- **默认值**: `50`
- **说明**: KNN 和 BM25 各自召回的候选数，也是 RRF 融合后送入 reranker 的候选上限。

| 取值 | 影响 |
| ---- | ---- |
| 过小 | reranker 之前候选不足，容易漏召回 |
| 适中 | 召回质量和延迟平衡 |
| 过大 | reranker 调用次数增加，延迟明显上升 |

### RERANKER_TOP_K

- **默认值**: `5`
- **说明**: reranker 精排后保留的片段数。生成回答时上下文主要来自这些片段。

| 取值 | 影响 |
| ---- | ---- |
| 过小 | 上下文不足，答案可能缺细节 |
| 适中 | 推荐 3-8 |
| 过大 | 噪声变多，LLM 更容易被无关片段干扰 |

---

## RRF 融合参数

### RRF_RANK_CONSTANT

- **默认值**: `60`
- **说明**: RRF 公式 `1 / (k + rank)` 中的常数。数值越大，前几名的优势越不明显。

### KNN_WEIGHT / BM25_WEIGHT

- **默认值**: `0.6` / `0.4`
- **约束**: 二者之和必须等于 `1.0`
- **说明**: 控制向量召回和 BM25 召回在 RRF 融合中的相对贡献。

| 现象 | 调整方向 |
| ---- | -------- |
| 原文关键词明确但搜不到 | 提高 `BM25_WEIGHT` |
| 问法偏概念化/同义改写 | 提高 `KNN_WEIGHT` |
| 结果语义相关但答非所问 | 降低 `KNN_WEIGHT` 或减少 `RERANKER_TOP_K` |

---

## 向量维度

### VECTOR_DIMS

- **默认值**: `1024`
- **说明**: sqlite-vec 表的向量维度，必须与 embedding 模型输出一致。

修改 `VECTOR_DIMS` 或 embedding 模型后，需要使用 `--rebuild` 重建索引，否则向量写入或检索会失败。

---

## 增量索引参数

### DATA_DIR

- **默认值**: `docs`
- **说明**: 文档根目录。索引器会递归扫描支持的文档格式，并用相对路径作为文档唯一标识。

增量入库会根据 `documents` 表中的 `source_path` 和 `file_hash` 跳过未变更文件；文件被删除后，下次入库会清理对应旧片段。

当前解析层位于 `src/parsers.py`：

| 类型 | 说明 |
| ---- | ---- |
| txt/text、PDF/Office/HTML/Markdown/图片等 | 通过 Docling 转 Markdown 后进入统一结构识别和切片流程 |

结构识别是解析后的公共后处理，会识别 Markdown 标题，以及 `第1章`、`第一节`、`第一编`、`第123条`、`卷一` 等中文结构。内部使用通用 `section/section_title/section_level` 元数据；当前数据库字段仍兼容写入 `chapter/chapter_title`。

音视频转写通常需要额外 ASR 依赖，当前不默认纳入目录扫描。

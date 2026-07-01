# RAG 知识库问答系统

基于 **sqlite-vec + FTS5 混合检索 + Docling 文档解析 + 规则切片 + Reranker 精排 + Ollama/本地模型后端** 的最小 RAG 实现，支持多格式文档增量入库。

## 架构

```
文档入库 → Docling/文本解析 → 通用结构识别 → 规则切片 → 向量 + BM25 双路索引
用户提问 → 混合检索（KNN + BM25, RRF 融合） → 精排 → LLM 生成回答
```

入库流程：

1. **文档解析** — 支持格式统一通过 Docling 转 Markdown
2. **结构识别** — 统一识别 Markdown 标题、中文章/节/编/条等 section 元数据
3. **文本切片** — 使用结构/规则切片，按段落/句子合并并保留 overlap
4. **KNN 向量索引** — sqlite-vec 余弦相似度语义匹配
5. **BM25 文本索引** — SQLite FTS5 全文匹配

检索流程：

1. **KNN 向量检索** — sqlite-vec 余弦相似度语义匹配
2. **BM25 文本检索** — SQLite FTS5 全文匹配
3. **RRF 融合排序** — 加权 Reciprocal Rank Fusion，合并两路结果
4. **Qwen3-Reranker 精排** — 对融合结果二次排序，提升最终命中率

## 项目结构

```
rag/
├── src/
│   ├── config.py          # 统一配置（支持 .env 覆盖）
│   ├── models.py          # 模型调用统一封装（embedding/reranker/LLM）
│   ├── parsers.py         # 文档解析 + 通用结构识别
│   ├── splitter.py        # 结构/规则切片
│   ├── match.py           # 混合检索 + RRF 融合
│   ├── indexer.py         # 索引构建（多格式、增量入库）
│   └── main.py            # 交互式问答入口
├── tests/
│   ├── eval_retrieval.py  # 检索质量评测脚本
│   ├── eval_dataset.json  # 评测数据集
│   └── debug_recall.py    # 召回调试工具
├── data/                  # SQLite 数据库文件（自动生成）
├── docs/                  # 知识库文档
├── .env                   # 运行时配置
├── .env.example           # 配置模板
└── pyproject.toml         # 项目依赖（uv 管理）
```

## 快速开始

### 1. 安装依赖

```bash
# 安装 uv（如已安装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 按需修改 .env 中的参数（默认值开箱即用）
```

### 3. 准备模型

```bash
# 默认使用 Ollama 作为 embedding/reranker/LLM 后端
ollama pull bge-large:latest
ollama pull qwen3-reranker-0.6b
ollama pull qwen2.5:3b

# 如果设置 EMBED_BACKEND=local / RERANKER_BACKEND=local，再下载本地模型：
# Qwen3-Embedding 默认路径: ~/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B
# 默认路径: ~/.cache/modelscope/hub/models/Qwen/Qwen3-Reranker-0___6B
```

### 4. 构建索引

```bash
# 扫描 docs/ 目录，增量入库（支持 txt、PDF、Office、HTML、图片等）
uv run python -m src.indexer

# 全量重建索引
uv run python -m src.indexer --rebuild
```

### 5. 交互式问答

```bash
uv run python src/main.py
```

运行后进入交互模式，输入问题回车即可获得回答，输入 `q` 退出。

## 配置说明

所有参数支持 `.env` 环境变量覆盖，默认值定义在 `src/config.py` 中。

| 参数                        | 默认值                                                    | 说明                         |
| --------------------------- | --------------------------------------------------------- | ---------------------------- |
| `DB_PATH`                   | `data/rag.db`                                             | SQLite 数据库路径            |
| `EMBED_BACKEND`             | `ollama`                                                  | 嵌入后端：ollama / local     |
| `RERANKER_BACKEND`          | `ollama`                                                  | 精排后端：ollama / local     |
| `LLM_BACKEND`               | `ollama`                                                  | LLM 后端：当前仅 ollama 可用 |
| `LOCAL_MODEL_DIR`           | `~/.cache/modelscope/.../Qwen`                            | 本地模型根目录               |
| `EMBED_MODEL_PATH`          | `~/.cache/modelscope/.../Qwen3-Embedding-0___6B`          | 本地嵌入模型路径             |
| `VECTOR_DIMS`               | `1024`                                                    | 向量维度                     |
| `RERANKER_MODEL_PATH`       | `~/.cache/modelscope/.../Qwen3-Reranker-0___6B`           | 本地精排模型路径             |
| `RERANKER_TOP_K`            | `5`                                                       | 精排后保留片段数             |
| `CHUNK_SIZE`                | `700`                                                     | 目标 chunk 长度              |
| `CHUNK_OVERLAP`             | `100`                                                     | 相邻 chunk 重叠字符数        |
| `OLLAMA_BASE_URL`           | `http://localhost:11434`                                  | Ollama 服务地址              |
| `CHAT_MODEL_NAME`           | `qwen2.5:3b`                                              | LLM 生成模型                 |
| `OLLAMA_EMBED_MODEL`        | `bge-large:latest`                                        | Ollama 嵌入模型              |
| `OLLAMA_RERANKER_MODEL`     | `qwen3-reranker-0.6b`                                     | Ollama 精排模型              |
| `DATA_DIR`                  | `docs`                                                    | 文档目录                     |
| `SEARCH_TOP_K`              | `15`                                                      | 检索返回片段数               |
| `RECALL_TOP_K`              | `50`                                                      | 精排前候选数                 |
| `RRF_RANK_CONSTANT`         | `60`                                                      | RRF 排名平滑常数             |
| `KNN_WEIGHT`                | `0.6`                                                     | KNN 向量检索权重             |
| `BM25_WEIGHT`               | `0.4`                                                     | BM25 文本检索权重            |

## 核心特性

### 文本切片

`src/splitter.py` 使用结构/规则切片：先按段落和句末标点拆分，再合并到 `CHUNK_SIZE` 附近，并用 `CHUNK_OVERLAP` 保留相邻片段上下文。切片阶段不调用 embedding，入库时只对最终 chunk 做一轮向量化。

当前推荐参数：

```env
CHUNK_SIZE=700
CHUNK_OVERLAP=100
OLLAMA_EMBED_MODEL=bge-large:latest
RECALL_TOP_K=50
KNN_WEIGHT=0.6
BM25_WEIGHT=0.4
```

### 已弃用：semantic 语义切片

早期 semantic 方案会先对句子或缓冲句做 embedding，根据相邻句向量距离寻找语义断点，再对最终 chunk 做 embedding 入库。这意味着重建索引时存在两轮向量化：一轮用于找切片边界，一轮用于写入向量索引。

在当前数据集实验中，semantic 方案约产生 `15162` 个 chunk，MRR 约 `0.518`；切回结构/规则切片后约产生 `1727` 个 chunk，评测结果为来源命中率 `80.0%`、答案片段命中率 `70.0%`、MRR `0.608`。因此 semantic 没有带来质量收益，反而显著放大重建耗时和索引规模，已从运行时代码和配置中移除。

### 可切换模型后端

嵌入和精排支持 `ollama` 与 `local` 两种后端。默认配置使用 Ollama API；如需完全本地加载嵌入/精排模型，可设置 `EMBED_BACKEND=local`、`RERANKER_BACKEND=local` 并准备对应模型目录。LLM 当前通过 Ollama chat API 调用。

### 多格式文档支持

文档解析和结构识别已从索引构建中解耦到 `src/parsers.py`。`src/indexer.py` 只负责扫描文件、增量判断、向量化和写库。

- Docling 解析：txt/text、PDF、DOCX、PPTX、XLSX、ODF、EPUB、HTML/XHTML、Markdown、AsciiDoc、LaTeX、CSV、常见图片、邮件等会先统一转为 Markdown。
- 通用后处理：统一识别 Markdown 标题，以及 `第1章`、`第一节`、`第一编`、`第123条`、`卷一` 等中文结构，写入 section 元数据；当前数据库字段仍兼容写入 `chapter/chapter_title`。
- 音视频转写通常需要额外 ASR 依赖，当前不默认纳入目录扫描。

### 增量入库

通过 `documents` 元数据表记录相对路径、MD5、mtime、文件大小和 chunk 数。未变更文件自动跳过；变更文件会先删除旧片段再重新入库；从文档目录删除的文件会在下次入库时清理旧索引。同名文件用相对路径区分。

### 混合检索 + RRF 融合 + 精排

`src/match.py` 实现两路独立检索后手动 RRF 融合，再送入 Qwen3-Reranker 精排：

```
recall = RRF(KNN + BM25, top_k=50)
final  = Reranker(recall, top_k=5)
```

### 测试与检索质量评测

快速单元测试不依赖真实模型或已有索引，覆盖切片、检索融合、索引元数据迁移和评测指标：

```bash
uv run python -m tests
```

检索质量评测会调用真实 embedding/reranker 和当前数据库索引：

```bash
uv run python -m tests.eval_retrieval
```

输出 MRR、来源命中率、答案片段命中率等指标。

## 技术栈

- **sqlite-vec** — 向量相似度检索（cosine distance）
- **SQLite FTS5** — BM25 全文检索
- **Embedding 模型** — 向量嵌入（Ollama 或本地 SentenceTransformer）
- **Reranker 模型** — 候选片段精排（Ollama 或本地 CrossEncoder）
- **Ollama** — 本地模型服务与 LLM 推理（qwen2.5:3b）
- **Docling** — 多格式文档解析与 Markdown 导出
- **uv** — Python 包管理

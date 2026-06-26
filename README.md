# RAG 知识库问答系统

基于 **Elasticsearch 混合检索 + Ollama** 的最小 RAG 实现，支持多格式文档（txt/pdf）增量入库。

## 架构

```
用户提问 → Ollama Embedding 向量化 → 混合检索（KNN + BM25, RRF 融合） → LLM 生成回答
```

检索流程：

1. **KNN 向量检索** — Ollama bge-large 编码查询，ES cosine similarity 语义匹配
2. **BM25 文本检索** — ES ik 分词全文匹配，弥补向量检索不足
3. **RRF 融合排序** — 加权 Reciprocal Rank Fusion，合并两路结果

## 项目结构

```
rag/
├── src/
│   ├── config.py          # 统一配置（支持 .env 覆盖）
│   ├── embedding.py       # 向量化（Ollama embedding API）
│   ├── llm.py             # LLM 生成（Ollama chat API）
│   ├── match.py           # 混合检索 + RRF 融合
│   ├── indexer.py         # 索引构建（多格式、增量入库）
│   └── main.py            # 交互式问答入口
├── tests/
│   ├── eval.py            # 检索质量评脚本
│   ├── eval_dataset.json  # 评测数据集
│   └── debug_recall.py    # 召回调试工具
├── docker/
│   ├── docker-compose.yml # Elasticsearch + IK 分词
│   └── Dockerfile.elasticsearch
├── docs/                  # 知识库文档（txt/pdf）
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

### 3. 启动 Elasticsearch

```bash
cd docker && docker compose up -d
# 等待 ES 就绪（约 30 秒）
curl http://localhost:9200/_cluster/health
```

### 3. 安装 Ollama 模型

```bash
# 安装 Ollama 后拉取所需模型
ollama pull bge-large        # 向量嵌入模型
ollama pull qwen2.5:3b       # LLM 生成模型
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 按需修改 .env 中的参数
```

### 5. 构建索引

```bash
# 扫描 docs/ 目录，增量入库（支持 txt/pdf）
uv run python -m src.indexer

# 全量重建索引
uv run python -m src.indexer --rebuild
```

### 6. 交互式问答

```bash
uv run python src/main.py
```

运行后进入交互模式，输入问题回车即可获得回答，输入 `q` 退出。

## 配置说明

所有参数支持 `.env` 环境变量覆盖，默认值定义在 `src/config.py` 中。

| 参数                    | 默认值                  | 说明                   |
| ----------------------- | ----------------------- | ---------------------- |
| `ES_URL`                | `http://localhost:9200` | Elasticsearch 地址     |
| `ES_INDEX_NAME`         | `hongloumeng_index`     | 索引名称               |
| `EMBED_MODEL_NAME`      | `bge-large:latest`      | Ollama 嵌入模型        |
| `CHAT_MODEL_NAME`       | `qwen2.5:3b`            | Ollama 生成模型        |
| `CHUNK_SIZE`            | `600`                   | 文本切片大小（字符数） |
| `CHUNK_OVERLAP`         | `80`                    | 切片重叠字符数         |
| `SEARCH_TOP_K`          | `5`                     | 检索返回片段数         |
| `SEARCH_NUM_CANDIDATES` | `200`                   | KNN 粗筛候选数         |
| `KNN_WEIGHT`            | `0.7`                   | KNN 向量检索权重       |
| `BM25_WEIGHT`           | `0.3`                   | BM25 文本检索权重      |

## 核心特性

### 多格式文档支持

`src/indexer.py` 支持 txt 和 pdf 格式，txt 文件自动识别章节结构按章分段切片，pdf 按页提取文本后切片。

### 增量入库

通过 MD5 哈希判断文件是否变更，未变更的文件自动跳过，变更的文件先删除旧文档再重新入库。

### 混合检索 + RRF 融合

`src/match.py` 实现两路独立检索后手动 RRF 融合：

```
score = KNN_WEIGHT * 1/(k+rank_knn) + BM25_WEIGHT * 1/(k+rank_bm25)
```

### 检索质量评测

```bash
uv run python -m tests.eval
```

输出 MRR、来源命中率、答案片段命中率等指标。

## 技术栈

- **Elasticsearch 8.x** + IK 分词 — 向量索引 + BM25 全文检索
- **Ollama** — 本地 LLM 推理 + 向量嵌入（bge-large / qwen2.5）
- **langchain-text-splitters** — 中文优先分隔符切片
- **PyMuPDF** — PDF 文本提取
- **uv** — Python 包管理

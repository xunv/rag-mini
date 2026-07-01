"""统一配置管理，支持 .env 环境变量覆盖"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 从项目根目录加载 .env（无论从哪里运行）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _resolve_path(value: str, base: Path = _PROJECT_ROOT) -> str:
    """将 .env 中的相对路径统一解析到项目根目录下"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


PROJECT_ROOT = str(_PROJECT_ROOT)

# ==================== SQLite 数据库 ====================
DB_PATH = _resolve_path(os.getenv("DB_PATH", "data/rag.db"))

# ==================== 模型后端选择 ====================
# 每个模型可选 "local"（本地加载）或 "ollama"（调用 Ollama API）
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "ollama")
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "ollama")
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# ==================== 本地模型根目录 ====================
LOCAL_MODEL_DIR = os.getenv(
    "LOCAL_MODEL_DIR",
    str(Path.home() / ".cache" / "modelscope" / "hub" / "models" / "Qwen"),
)
LOCAL_MODEL_DIR = _resolve_path(LOCAL_MODEL_DIR)

# ==================== 嵌入模型（本地 Qwen3-Embedding） ====================
EMBED_MODEL_PATH = os.getenv(
    "EMBED_MODEL_PATH",
    str(Path(LOCAL_MODEL_DIR) / "Qwen3-Embedding-0___6B"),
)
EMBED_MODEL_PATH = _resolve_path(EMBED_MODEL_PATH)
VECTOR_DIMS = int(os.getenv("VECTOR_DIMS", "1024"))

# ==================== 精排模型（本地 Qwen3-Reranker） ====================
RERANKER_MODEL_PATH = os.getenv(
    "RERANKER_MODEL_PATH",
    str(Path(LOCAL_MODEL_DIR) / "Qwen3-Reranker-0___6B"),
)
RERANKER_MODEL_PATH = _resolve_path(RERANKER_MODEL_PATH)
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))

# ==================== 切片参数 ====================
# 结构/规则切片，只在入库向量化时对最终 chunk 调用 embedding
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ==================== Ollama（仅用于 LLM 生成） ====================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "qwen2.5:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-large:latest")
OLLAMA_RERANKER_MODEL = os.getenv("OLLAMA_RERANKER_MODEL", "qwen3-reranker-0.6b")

# ==================== 数据文件 ====================
DATA_DIR = _resolve_path(os.getenv("DATA_DIR", "docs"))

# ==================== 检索参数 ====================
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "15"))
# 精排前的候选数（RRF融合后取该数量送入reranker）
RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", "50"))

# ==================== RRF 融合参数 ====================
RRF_RANK_CONSTANT = int(os.getenv("RRF_RANK_CONSTANT", "60"))
KNN_WEIGHT = float(os.getenv("KNN_WEIGHT", "0.6"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))

# ==================== 参数校验 ====================
_valid_backends = {"local", "ollama"}
for _name, _backend in {
    "EMBED_BACKEND": EMBED_BACKEND,
    "RERANKER_BACKEND": RERANKER_BACKEND,
    "LLM_BACKEND": LLM_BACKEND,
}.items():
    if _backend not in _valid_backends:
        raise ValueError(f"{_name} 必须是 local 或 ollama，当前为 {_backend}")

if CHUNK_SIZE < 50:
    raise ValueError(f"CHUNK_SIZE ({CHUNK_SIZE}) 必须 >= 50")

if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) 必须 >= 0 且小于 CHUNK_SIZE ({CHUNK_SIZE})")

_weight_sum = KNN_WEIGHT + BM25_WEIGHT
if abs(_weight_sum - 1.0) > 1e-6:
    raise ValueError(f"KNN_WEIGHT + BM25_WEIGHT 必须等于 1.0，当前为 {_weight_sum}")

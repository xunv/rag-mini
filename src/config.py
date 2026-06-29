"""统一配置管理，支持 .env 环境变量覆盖"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 从项目根目录加载 .env（无论从哪里运行）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ==================== SQLite 数据库 ====================
DB_PATH = os.getenv("DB_PATH", str(_PROJECT_ROOT / "data" / "rag.db"))

# ==================== 嵌入模型（通过 Ollama embedding API） ====================
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "bge-large:latest")
VECTOR_DIMS = int(os.getenv("VECTOR_DIMS", "1024"))

# ==================== Ollama（仅用于 LLM 生成） ====================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "qwen2.5:3b")

# ==================== 数据文件 ====================
DATA_DIR = os.getenv("DATA_DIR", "docs")

# ==================== 切片参数 ====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
# 中文优先分隔符
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]

# ==================== 检索参数 ====================
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "15"))

# ==================== RRF 融合参数 ====================
RRF_RANK_CONSTANT = int(os.getenv("RRF_RANK_CONSTANT", "60"))
KNN_WEIGHT = float(os.getenv("KNN_WEIGHT", "0.25"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.75"))

# ==================== 参数校验 ====================
_weight_sum = KNN_WEIGHT + BM25_WEIGHT
if abs(_weight_sum - 1.0) > 1e-6:
    raise ValueError(f"KNN_WEIGHT + BM25_WEIGHT 必须等于 1.0，当前为 {_weight_sum}")

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) 必须小于 CHUNK_SIZE ({CHUNK_SIZE})")
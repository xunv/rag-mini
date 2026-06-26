from src.match import _knn_search, _bm25_search, get_es_client
from src.embedding import get_embedding
from src.config import SEARCH_NUM_CANDIDATES

question = '贾宝玉的通灵宝玉上刻着什么字？'
query_vector = get_embedding(question)
client = get_es_client()

# 分别查看 KNN 和 BM25 的召回情况
knn_results = _knn_search(client, query_vector, 45, SEARCH_NUM_CANDIDATES)
bm25_results = _bm25_search(client, question, 45)

print('=== KNN Top-10 ===')
for i, r in enumerate(knn_results[:10], 1):
    print(f'{i}. {r["chapter"]}')

print()
print('=== BM25 Top-10 ===')
for i, r in enumerate(bm25_results[:10], 1):
    print(f'{i}. {r["chapter"]}')

# 检查第8章包含关键词的切片排名
print()
print('=== 第8章在 KNN 中的排名 ===')
for i, r in enumerate(knn_results, 1):
    if r['chapter'] == '第8章':
        has_keyword = '莫失莫忘' in r['content']
        print(f'KNN Rank {i}: 包含关键词={has_keyword}')

print()
print('=== 第8章在 BM25 中的排名 ===')
for i, r in enumerate(bm25_results, 1):
    if r['chapter'] == '第8章':
        has_keyword = '莫失莫忘' in r['content']
        print(f'BM25 Rank {i}: 包含关键词={has_keyword}')
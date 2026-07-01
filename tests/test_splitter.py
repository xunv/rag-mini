import unittest

from src import splitter


class SplitterTest(unittest.TestCase):
    def test_comma_delimiter_splits_chinese_and_ascii_commas(self):
        text = "甲，乙,丙。"

        result = splitter._COMMA_DELIM.sub(r"\1\2\n", text)

        self.assertEqual(result, "甲，\n乙,\n丙。")

    def test_split_text_keeps_chunk_size_near_target(self):
        text = (
            "第一句内容已经足够长用于测试。第二句内容已经足够长用于测试。"
            "第三句内容已经足够长用于测试。第四句内容已经足够长用于测试。"
        )

        chunks = splitter.split_text(text, chunk_size=45, chunk_overlap=10)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 55 for chunk in chunks))

    def test_split_text_adds_overlap_between_chunks(self):
        text = "甲乙丙丁戊己庚辛壬癸。子丑寅卯辰巳午未申酉。天地玄黄宇宙洪荒日月盈昃。"

        chunks = splitter.split_text(text, chunk_size=22, chunk_overlap=5)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[1].startswith(chunks[0][-5:]))

    def test_split_text_splits_long_sentence_by_commas(self):
        text = "甲乙丙丁戊己庚辛壬癸，子丑寅卯辰巳午未申酉,天地玄黄宇宙洪荒日月盈昃。"

        chunks = splitter.split_text(text, chunk_size=20, chunk_overlap=0)

        self.assertEqual(
            chunks,
            ["甲乙丙丁戊己庚辛壬癸，", "子丑寅卯辰巳午未申酉,", "天地玄黄宇宙洪荒日月盈昃。"],
        )

    def test_splitter_has_no_embedding_dependency(self):
        self.assertFalse(hasattr(splitter, "get_embeddings_batch"))


if __name__ == "__main__":
    unittest.main()

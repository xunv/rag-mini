import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src import parsers


class ParserTest(unittest.TestCase):
    def test_parse_txt_uses_docling_and_common_section_metadata(self):
        result = Mock()
        result.document.export_to_markdown.return_value = (
            "第1章 开端\n第一句内容已经足够长用于测试。第二句内容已经足够长用于测试。"
        )
        converter = Mock()
        converter.convert.return_value = result

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "book.txt"
            path.write_text("raw text", encoding="utf-8")
            with patch.object(parsers, "_get_docling_converter", return_value=converter):
                chunks = parsers.parse_file(str(path))

        converter.convert.assert_called_once_with(path)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section"], "第1章")
        self.assertEqual(chunks[0]["section_title"], "开端")
        self.assertEqual(chunks[0]["chapter"], "第1章")
        self.assertEqual(chunks[0]["chapter_title"], "开端")
        self.assertIn("第一句内容", chunks[0]["text"])

    def test_parse_docling_format_exports_markdown_then_splits(self):
        result = Mock()
        result.document.export_to_markdown.return_value = (
            "## 第2章 转折\n第一句内容已经足够长用于测试。第二句内容已经足够长用于测试。"
        )
        converter = Mock()
        converter.convert.return_value = result

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.pdf"
            path.write_bytes(b"%PDF-1.4")
            with patch.object(parsers, "_get_docling_converter", return_value=converter):
                chunks = parsers.parse_file(str(path))

        converter.convert.assert_called_once_with(path)
        self.assertEqual(chunks[0]["section"], "第2章")
        self.assertEqual(chunks[0]["section_title"], "转折")
        self.assertEqual(chunks[0]["section_level"], 2)
        self.assertEqual(chunks[0]["chapter"], "第2章")
        self.assertIn("第二句内容", chunks[0]["text"])

    def test_markdown_heading_without_chinese_number_becomes_section(self):
        chunks = parsers._split_text_with_sections(
            "# Overview\n第一句内容已经足够长用于测试。第二句内容已经足够长用于测试。"
        )

        self.assertEqual(chunks[0]["section"], "Overview")
        self.assertEqual(chunks[0]["section_title"], "")
        self.assertEqual(chunks[0]["section_level"], 1)

    def test_parse_file_rejects_unsupported_extension(self):
        with self.assertRaisesRegex(ValueError, "不支持的文件格式"):
            parsers.parse_file("archive.zip")

    def test_supported_extensions_include_common_office_formats(self):
        self.assertIn(".pdf", parsers.SUPPORTED_EXTENSIONS)
        self.assertIn(".docx", parsers.SUPPORTED_EXTENSIONS)
        self.assertIn(".pptx", parsers.SUPPORTED_EXTENSIONS)
        self.assertIn(".xlsx", parsers.SUPPORTED_EXTENSIONS)


if __name__ == "__main__":
    unittest.main()

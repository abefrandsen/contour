import tempfile
from pathlib import Path
import unittest

from contour.mdio import (
    FrontmatterError,
    extract_wikilinks,
    parse_markdown,
    parse_markdown_file,
    strip_wiki,
    write_markdown_file,
)


class TestMdio(unittest.TestCase):
    def test_strip_wiki_aliases_and_headings(self) -> None:
        self.assertEqual(strip_wiki("[[run_001|alias]]"), "run_001")
        self.assertEqual(strip_wiki("[[clm_x#section]]"), "clm_x")
        self.assertEqual(strip_wiki("plain_id"), "plain_id")

    def test_extract_wikilinks(self) -> None:
        links = extract_wikilinks("See [[run_a|first]] and [[run_b#obs]] in body.")
        self.assertEqual(links, {"run_a", "run_b"})

    def test_crlf_and_empty_frontmatter(self) -> None:
        fm, body = parse_markdown("---\r\n---\r\n\r\nHello\r\n")
        self.assertEqual(fm, {})
        self.assertEqual(body, "\nHello\n")

    def test_malformed_yaml_raises(self) -> None:
        with self.assertRaises(FrontmatterError):
            parse_markdown("---\nfoo: [unclosed\n---\nbody\n")

    def test_roundtrip_quotes_in_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "node.md"
            write_markdown_file(p, {"id": "clm_q", "title": 'Title with "quotes"'}, "## Body\n- ok\n")
            fm, body = parse_markdown_file(p)
            self.assertEqual(fm["title"], 'Title with "quotes"')
            self.assertIn("- ok", body)


if __name__ == "__main__":
    unittest.main()

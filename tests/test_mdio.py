import unittest

from contour.mdio import (
    FrontmatterError,
    parse_markdown,
    render_markdown,
    strip_wiki,
    strip_wiki_list,
)


class TestMdio(unittest.TestCase):
    def test_strip_wiki_aliases_and_headings(self) -> None:
        self.assertEqual(strip_wiki("[[crn_foo|My Alias]]"), "crn_foo")
        self.assertEqual(strip_wiki("[[crn_foo#Section]]"), "crn_foo")
        self.assertEqual(strip_wiki("  crn_foo  "), "crn_foo")
        self.assertEqual(
            strip_wiki_list(["[[a|1]]", "b#x"]),
            ["a", "b"],
        )

    def test_crlf_and_empty_frontmatter(self) -> None:
        fm, body = parse_markdown("---\r\n---\r\n\r\nHello\r\n")
        self.assertEqual(fm, {})
        self.assertEqual(body, "\nHello\n")

    def test_malformed_yaml_raises(self) -> None:
        with self.assertRaises(FrontmatterError):
            parse_markdown("---\ntitle: [unclosed\n---\nBody\n")

    def test_roundtrip_quotes_in_title(self) -> None:
        md = render_markdown({"id": "crn_x", "title": 'Has "quotes" & : colons'}, "Body")
        fm, body = parse_markdown(md)
        self.assertEqual(fm["title"], 'Has "quotes" & : colons')
        self.assertEqual(body.strip(), "Body")


if __name__ == "__main__":
    unittest.main()

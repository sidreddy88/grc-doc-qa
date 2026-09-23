from app.services.pdf_structure import (
    Heading,
    find_running_header,
    locate_headings,
    looks_like_toc_page,
    parse_toc_lines,
    split_at_headings,
    strip_page_furniture,
)
from app.services.text import fold_for_match

HEADER = "Acme SOC 2 Report on Controls"


def test_running_header_detected_across_pages():
    bodies = ["Encryption is enforced", "Access reviews run", "Backups are tested", "Vendors are assessed"]
    pages = [f"{HEADER} {body} quarterly." for body in bodies]
    assert find_running_header(pages) == HEADER


def test_no_header_when_pages_differ():
    assert find_running_header(["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]) is None


def test_strip_page_furniture_removes_header_and_page_number():
    raw = f"{HEADER}\n  Real   content here.\n 17 "
    assert strip_page_furniture(raw, HEADER, 17) == "Real content here."


def test_strip_page_furniture_keeps_numbers_that_are_not_the_page():
    assert strip_page_furniture("Retention is 90", None, 17) == "Retention is 90"


def test_parse_toc_joins_wrapped_titles_and_ignores_marker():
    layout = """
        Table of Contents
             Management's Assertion                                  5
             DC 5: The applicable trust services criteria and the
             related controls                                       17
                 5.1 Integrity and Ethical Values                  17
             DC 9: Disclosures of significant changes in last 1 year   26
    """
    assert parse_toc_lines(layout) == [
        "Management's Assertion",
        "DC 5: The applicable trust services criteria and the related controls",
        "5.1 Integrity and Ethical Values",
        "DC 9: Disclosures of significant changes in last 1 year",
    ]


def test_parse_toc_supports_dot_leaders():
    assert parse_toc_lines("Scope ........ 8\nOpinion.....10") == ["Scope", "Opinion"]


def test_looks_like_toc_page():
    toc = "\n".join(f"Section {i}      {i + 3}" for i in range(8))
    prose = "\n".join("This is a normal paragraph of body text without numbers" for _ in range(8))
    assert looks_like_toc_page(toc)
    assert not looks_like_toc_page(prose)


def test_locate_headings_in_order_with_quote_folding():
    pages = [
        (6, "Management’s Assertion We have prepared the description."),
        (9, "Scope We examined. Opinion In our opinion the controls operated."),
    ]
    headings = locate_headings(pages, ["Management's Assertion", "Scope", "Opinion"])
    assert [(h.page, h.title) for h in headings] == [(6, "Management's Assertion"), (9, "Scope"), (9, "Opinion")]


def test_locate_headings_skips_titles_that_never_appear():
    pages = [(1, "Intro text. Beta section starts. Gamma follows.")]
    headings = locate_headings(pages, ["Alpha", "Beta", "Gamma"])
    assert [h.title for h in headings] == ["Beta", "Gamma"]


def test_locate_headings_requires_word_boundaries():
    pages = [(1, "Telescope readings. Scope of work.")]
    headings = locate_headings(pages, ["Scope"])
    assert headings[0].offset == fold_for_match(pages[0][1]).index("scope of")


def test_split_at_headings_carries_section_across_pages():
    pages = [(1, "Preamble. Alpha Body one."), (2, "More alpha body. Beta Body two.")]
    headings = [Heading(page=1, offset=10, title="Alpha"), Heading(page=2, offset=17, title="Beta")]
    assert split_at_headings(pages, headings) == [
        (1, None, "Preamble."),
        (1, "Alpha", "Alpha Body one."),
        (2, "Alpha", "More alpha body."),
        (2, "Beta", "Beta Body two."),
    ]

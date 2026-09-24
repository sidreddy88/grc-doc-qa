import re
from pathlib import Path

from app.core.config import Settings
from app.models import DocumentType
from app.services.chunking import chunk_document
from app.services.control_matrix import Word, partition_matrix, rows_from_layout, rows_from_text
from app.services.control_matrix import MatrixRow
from app.services.document_loader import LoadedDocument, Segment, load_document

CC23 = "CC2.3 The entity communicates with external parties regarding matters affecting the functioning of internal control."
CC72 = "CC7.2 The entity monitors system components for anomalies."
PORTAL = ("Nave provides a support portal to customers to report incidents. "
          "Inspected the Contact Us Page to determine that a portal is provided. No exceptions noted.")
IR_PLAN = ("Nave has a documented incident response plan. "
           "Inspected the Incident Response Plan to determine that roles are defined. No exceptions noted.")
ALERTS = "Nave uses monitoring alerts. Inspected alert configuration to determine that alerts exist. No exceptions noted."


def _segment(text: str, page: int, section: str = "Test Results") -> Segment:
    return Segment(text=text, page=page, section=section)


def test_rows_are_split_and_criterion_wording_is_separated_from_the_control():
    matrix = _segment(f"{CC23} {PORTAL} {CC23} {IR_PLAN} {CC72} {ALERTS} {CC72} {PORTAL}", page=41)
    prose, matrix_segments = partition_matrix([matrix])
    assert prose == []
    rows = rows_from_text(matrix_segments)
    bodies = {row.body: row for row in rows}
    assert set(bodies) == {PORTAL, IR_PLAN, ALERTS}
    assert bodies[IR_PLAN].criteria == ["CC2.3"]
    assert bodies[IR_PLAN].index_context == CC23


def test_identical_controls_under_several_criteria_become_one_row():
    matrix = _segment(f"{CC23} {PORTAL} {CC23} {IR_PLAN} {CC72} {ALERTS} {CC72} {PORTAL}", page=41)
    rows = rows_from_text(partition_matrix([matrix])[1])
    [portal] = [row for row in rows if row.body == PORTAL]
    assert portal.criteria == ["CC2.3", "CC7.2"]
    assert portal.section == "Test Results: CC2.3, CC7.2"
    assert portal.page == 41  # cited at its first occurrence


def test_spill_over_at_the_top_of_a_page_is_reattached_to_the_previous_row():
    first = _segment(f"{CC23} {PORTAL} {CC72} {ALERTS} {CC23} Nave has a documented incident Inspected the Incident",
                     page=48)
    second = _segment(f"response plan. Response Plan to determine that roles are defined. No exceptions noted. {CC72} {ALERTS} "
                      f"{CC72} {PORTAL}", page=49)
    rows = rows_from_text(partition_matrix([first, second])[1])
    [cut] = [row for row in rows if "documented incident" in row.body]
    assert cut.page == 48
    assert cut.body.endswith("response plan. Response Plan to determine that roles are defined. No exceptions noted.")
    assert not any(row.body.startswith("response plan") for row in rows)


def test_prose_and_the_matrix_introduction_stay_prose():
    intro = "Test Results The results of each test are listed alongside each test in the Testing Matrices."
    segments = [
        _segment("5.12 Incident Management Nave will inform all necessary parties without undue delay.", 21, "5.12"),
        _segment(f"{intro} {CC23} {PORTAL} {CC72} {ALERTS}", page=30),
    ]
    prose, matrix_segments = partition_matrix(segments)
    assert [p.text for p in prose] == [segments[0].text, intro]
    assert len(rows_from_text(matrix_segments)) == 2


def test_chunks_index_the_criterion_wording_but_do_not_display_it():
    row = MatrixRow(["CC2.3"], IR_PLAN, 42, [CC23])
    document = LoadedDocument(doc_type=DocumentType.PDF, segments=[], unit_count=1, matrix_rows=[row])
    [chunk] = chunk_document(document, Settings(_env_file=None))
    assert (chunk.text, chunk.page, chunk.section) == (IR_PLAN, 42, "Test Results: CC2.3")
    assert "communicates with external parties" in chunk.index_text
    assert "communicates with external parties" not in chunk.text


def _words(y: float, cells: dict[float, str]) -> list[Word]:
    """One table line: column x -> the words that line holds in that column."""
    words = []
    for x, text in cells.items():
        for i, token in enumerate(text.split()):
            words.append(Word(x=x + i * 30, y=y, text=token))
    return words


def test_layout_rebuilds_each_cell_of_a_row_split_across_pages():
    header = {36.0: "Acme Type 2 Report"}
    columns = (6.0, 58.0, 154.0, 312.0, 528.0)

    def line(y, *cells):
        return _words(y, dict(zip(columns, cells, strict=False)))

    page_a = [*_words(760, header),
              *line(700, "CC2.3", "The entity communicates", "Nave provides a portal.", "Inspected the portal.", "No exceptions noted."),
              *line(650, "CC7.2", "The entity monitors", "Nave has an incident", "Inspected the incident", "No exceptions noted."),
              *line(640, "", "", "response", "response plan")]
    page_b = [*_words(760, header),
              *line(710, "", "systems.", "plan.", "and its roles."),
              *line(690, "CC7.3", "The entity evaluates events.", "Nave triages events.", "Inspected the log.", "Not tested."),
              *_words(20, {300.0: "49"})]
    page_c = [*_words(760, header),
              *line(700, "CC8.1", "The entity manages change.", "Nave reviews changes.", "Inspected tickets.", "No exceptions noted.")]
    rows = rows_from_layout([(48, page_a), (49, page_b), (50, page_c)])
    by_id = {row.criteria[0]: row for row in rows}
    assert by_id["CC7.2"].body == "Nave has an incident response plan. Inspected the incident response plan and its roles. No exceptions noted."
    assert by_id["CC7.2"].criterion_texts == ["CC7.2 The entity monitors systems."]
    assert by_id["CC7.2"].page == 48
    assert set(by_id) == {"CC2.3", "CC7.2", "CC7.3", "CC8.1"}  # the header and page number are not rows


def test_layout_that_is_not_a_five_column_table_falls_back():
    words = _words(700, {58.0: "CC2.3 The entity communicates", 300.0: "No exceptions noted."})
    assert rows_from_layout([(40, words), (41, words), (42, words)]) == []


def test_json_documents_are_not_treated_as_matrices():
    document = LoadedDocument(
        doc_type=DocumentType.JSON, segments=[Segment(text=f"{CC23} {PORTAL} {CC72} {ALERTS}", source_id="r1")], unit_count=1
    )
    [chunk] = chunk_document(document, Settings(_env_file=None))
    assert chunk.source_id == "r1"
    assert chunk.section is None


def test_sample_report_matrix_has_one_control_per_chunk():
    settings = Settings(_env_file=None)
    pdf = (Path(__file__).parent.parent / "fixtures" / "nave_soc2.pdf").read_bytes()
    chunks = chunk_document(load_document(pdf, DocumentType.PDF, settings), settings)
    matrix = [c for c in chunks if (c.section or "").startswith("Test Results:")]
    assert len(matrix) >= 90  # ~245 table rows, heavily duplicated across criteria
    assert all(" The " in c.index_context or " The entity" in c.index_context for c in matrix)  # wording recovered
    # Old fixed-size windows mixed several rows per chunk; a row now carries exactly one test result.
    assert all(len(re.findall(r"No exceptions noted\.|Not tested\.", c.text)) <= 1 for c in matrix)
    [ir_plan] = [c for c in matrix if c.text.startswith("Nave has a documented incident response plan")]
    # CC4.2 and CC7.4's copies straddle page breaks; rebuilt cell by cell, they merge with the others.
    assert {"CC2.2", "CC2.3", "CC4.2", "CC7.3", "CC7.4"} <= set(ir_plan.section.removeprefix("Test Results: ").split(", "))

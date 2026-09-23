from app.models import Chunk
from app.services.faithfulness import Claim, build_judge_prompt, locate_quote, verify_citations
from app.services.qa_chain import DraftCitation

CHUNK_TEXT = (
    "Nave’s application runs in the Google Cloud Platform (GCP) utilizing Virtual Machines, Storage and "
    "Database services. Data is persisted in GCP Storage and utilizes Advanced Encryption Standard (“AES”) "
    "256 encrypted disks for all data stored at rest."
)


def test_exact_quote_returns_document_span():
    quote = "utilizes Advanced Encryption Standard"
    assert locate_quote(quote, CHUNK_TEXT) == quote


def test_folding_tolerates_case_curly_quotes_and_whitespace():
    quote = "nave's application   runs in the google cloud platform"
    span = locate_quote(quote, CHUNK_TEXT)
    assert span == "Nave’s application runs in the Google Cloud Platform"  # original characters returned


def test_fuzzy_match_tolerates_a_small_copying_slip():
    quote = "Data is persisted in GCP Storage and uses Advanced Encryption Standard"
    span = locate_quote(quote, CHUNK_TEXT)
    assert span is not None and span.startswith("Data is persisted in GCP Storage")


def test_fabricated_quote_is_rejected():
    assert locate_quote("Data is replicated to AWS us-east-1 every hour", CHUNK_TEXT) is None


def test_too_short_quote_is_rejected():
    assert locate_quote("GCP", CHUNK_TEXT) is None


def test_scattered_fragments_are_not_a_quote():
    quote = "Nave’s application ... all data stored at rest"
    assert locate_quote(quote, CHUNK_TEXT) is None


def test_multiline_kb_rows_match_across_field_breaks():
    text = "question: Where is data stored?\nanswer: US Central region on GCP."
    assert locate_quote("Where is data stored? answer: US Central", text) == "Where is data stored?\nanswer: US Central"


def test_verify_citations_drops_unverifiable_and_dedupes():
    chunk = Chunk(chunk_id=3, text=CHUNK_TEXT, page=16, section="3.3 Security Processes and Procedures")
    verified = verify_citations(
        [
            DraftCitation(chunk=chunk, quote="256 encrypted disks for all data stored at rest"),
            DraftCitation(chunk=chunk, quote="256 encrypted disks for all data stored at rest"),
            DraftCitation(chunk=chunk, quote="MFA is enforced for every employee login"),
        ]
    )
    assert [(v.chunk.page, v.excerpt) for v in verified] == [(16, "256 encrypted disks for all data stored at rest")]


def test_judge_prompt_lists_each_claim_with_its_evidence():
    prompt = build_judge_prompt(
        "Is data encrypted?",
        [Claim(claim_id="answer", text="Yes, AES-256.", evidence=["AES 256 encrypted disks"])],
    )
    assert "claim_id: answer" in prompt
    assert "  - AES 256 encrypted disks" in prompt

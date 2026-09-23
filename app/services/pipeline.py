from app.models import NOT_FOUND_ANSWER, AnswerResult, DocumentType


class QAPipeline:
    async def answer(self, questions: list[str], document: bytes, doc_type: DocumentType) -> list[AnswerResult]:
        # Placeholder until retrieval and synthesis land; keeps the endpoint contract testable.
        return [AnswerResult(question=question, answer=NOT_FOUND_ANSWER) for question in questions]

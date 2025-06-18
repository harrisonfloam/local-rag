from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    content: str = Field(..., description="The text content of the document chunk")
    score: float = Field(..., description="Relevance score (0.0 to 1.0)")
    document_id: str = Field(..., description="ID of the source document")
    document_title: str = Field(
        ..., description="Title/filename of the source document"
    )
    chunk_index: int = Field(..., description="Index of this chunk within the document")

    def __str__(self) -> str:
        return f"""Source: {self.document_title}\nScore: {self.score:.2f})\n{self.content}"""

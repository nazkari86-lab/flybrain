"""Validated declarations for immutable source datasets."""

from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl, model_validator


class Artifact(BaseModel):
    """One immutable downloadable source artifact."""

    url: HttpUrl
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceManifest(BaseModel):
    """Provenance and integrity information for a source dataset."""

    dataset_id: str = Field(min_length=1)
    source_publication: HttpUrl
    license: str = Field(min_length=1)
    artifacts: tuple[Artifact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_urls(self) -> "SourceManifest":
        urls = [str(artifact.url) for artifact in self.artifacts]
        if len(urls) != len(set(urls)):
            raise ValueError("duplicate artifact URL")
        return self


def load_manifest(path: Path) -> SourceManifest:
    """Load and validate a JSON source manifest."""

    return SourceManifest.model_validate_json(path.read_text())

"""Raw Amazon Musical Instruments table schemas from table_schema.md."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

RAW_REVIEW_FILE = "data/amazon_musical_instruments/Musical_Instruments.jsonl"
PRODUCT_METADATA_FILE = "data/amazon_musical_instruments/meta_Musical_Instruments.jsonl"


@dataclass(frozen=True, slots=True)
class ReviewImage:
    """One user-uploaded image attached to a raw review."""

    __file__: ClassVar[str] = RAW_REVIEW_FILE

    small_image_url: str | None
    medium_image_url: str | None
    large_image_url: str | None
    attachment_type: str


@dataclass(frozen=True, slots=True)
class RawReview:
    """One row from the raw review file (section 5.1)."""

    __file__: ClassVar[str] = RAW_REVIEW_FILE

    rating: float
    title: str | None
    text: str | None
    images: tuple[ReviewImage, ...]
    asin: str
    parent_asin: str
    user_id: str
    timestamp: int
    helpful_vote: int
    verified_purchase: bool


@dataclass(frozen=True, slots=True)
class ProductImage:
    """One product image from the metadata file."""

    __file__: ClassVar[str] = PRODUCT_METADATA_FILE

    thumb: str | None
    large: str | None
    hi_res: str | None
    variant: str | None


@dataclass(frozen=True, slots=True)
class ProductVideo:
    """One product video from the metadata file."""

    __file__: ClassVar[str] = PRODUCT_METADATA_FILE

    title: str | None
    url: str | None
    user_id: str | None


@dataclass(frozen=True, slots=True)
class ProductMetadata:
    """One row from the product metadata file (section 5.2)."""

    __file__: ClassVar[str] = PRODUCT_METADATA_FILE

    main_category: str | None
    title: str | None
    average_rating: float | None
    rating_number: int | None
    features: tuple[str, ...]
    description: tuple[str, ...]
    price: float | str | None
    images: tuple[ProductImage, ...]
    videos: tuple[ProductVideo, ...]
    store: str | None
    categories: tuple[str, ...]
    details: dict[str, Any]
    parent_asin: str
    bought_together: tuple[Any, ...] | None
    subtitle: str | None
    author: str | dict[str, Any] | None

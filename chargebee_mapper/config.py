"""Configuration loading and validation."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    api_key: str
    site: str
    max_concurrency: int = 20
    page_size: int = 100
    max_retries: int = 5
    rate_limit_rpm: int = 150
    log_level: str = "INFO"
    output_dir: Path = field(default_factory=lambda: Path("output"))

    @property
    def base_url(self) -> str:
        return f"https://{self.site}.chargebee.com/api/v2"

    @property
    def log_file(self) -> Path:
        return self.output_dir / "chargebee_mapper.log"


def load_config() -> Config:
    """Load configuration from environment variables (.env file supported)."""
    load_dotenv()

    api_key = os.environ.get("CHARGEBEE_API_KEY", "").strip()
    site = os.environ.get("CHARGEBEE_SITE", "").strip()

    if not api_key:
        print("Error: CHARGEBEE_API_KEY environment variable is not set.")
        print("Set it in a .env file or export it in your shell.")
        print("See .env.example for the expected format.")
        sys.exit(1)

    if not site:
        print("Error: CHARGEBEE_SITE environment variable is not set.")
        print("Set it in a .env file or export it in your shell.")
        print("See .env.example for the expected format.")
        sys.exit(1)

    return Config(
        api_key=api_key,
        site=site,
        max_concurrency=int(os.environ.get("CHARGEBEE_MAX_CONCURRENCY", "20")),
        page_size=min(int(os.environ.get("CHARGEBEE_PAGE_SIZE", "100")), 100),
        max_retries=int(os.environ.get("CHARGEBEE_MAX_RETRIES", "5")),
        rate_limit_rpm=int(os.environ.get("CHARGEBEE_RATE_LIMIT_RPM", "150")),
        log_level=os.environ.get("CHARGEBEE_LOG_LEVEL", "INFO").upper(),
        output_dir=Path(os.environ.get("CHARGEBEE_OUTPUT_DIR", "output")),
    )

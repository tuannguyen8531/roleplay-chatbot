"""Run the API server with configured defaults."""

import uvicorn

from app.config import config


def main():
    """Start Uvicorn using API_HOST/API_PORT from config."""
    uvicorn.run(
        "app.api.app:app",
        host=config.api_host,
        port=config.api_port,
    )


if __name__ == "__main__":
    main()

"""FastAPI server entry-point for the CodeDebtRefactor environment."""

import uvicorn

from openenv.core.env_server import create_fastapi_app

from server.environment import CodeDebtEnvironment
from models import FixCodeAction, TerminalObservation


def main() -> None:
    """Create the FastAPI application and start serving on port 7860."""
    app = create_fastapi_app(
        CodeDebtEnvironment,
        action_cls=FixCodeAction,
        observation_cls=TerminalObservation,
    )
    # REQUIRED for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
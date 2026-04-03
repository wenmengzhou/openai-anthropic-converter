#!/usr/bin/env bash
# Start the OpenAI-compatible server (forwards to Anthropic backend)
#
# Configuration is read from .env file automatically.
# Override with environment variables or CLI args:
#   ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY, --port, --log-level, etc.
#
# Usage:
#   ./start_openai_server.sh
#   ./start_openai_server.sh --port 8001 --log-level debug

set -euo pipefail
cd "$(dirname "$0")"

python -m openai_anthropic_converter.servers.openai_server --log-level debug "$@"

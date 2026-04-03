#!/usr/bin/env bash
# Start the Anthropic-compatible server (forwards to OpenAI backend)
#
# Configuration is read from .env file automatically.
# Override with environment variables or CLI args:
#   OPENAI_BASE_URL, OPENAI_API_KEY, --port, --log-level, etc.
#
# Usage:
#   ./start_anthropic_server.sh
#   ./start_anthropic_server.sh --port 8002 --log-level debug

set -euo pipefail
cd "$(dirname "$0")"

python -m openai_anthropic_converter.servers.anthropic_server --log-level debug "$@"

#!/bin/bash
# Double-click (macOS) or run ./start.command: builds, starts the app and opens the browser.
cd "$(dirname "$0")"
docker compose up -d --build || { echo "Is Docker Desktop running?"; read -n1 -p "Press any key"; exit 1; }
echo "Waiting for the app..."
until curl -fs http://localhost:8600/_stcore/health >/dev/null; do sleep 1; done
open http://localhost:8600 2>/dev/null || xdg-open http://localhost:8600

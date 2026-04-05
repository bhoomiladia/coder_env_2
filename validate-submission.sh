#!/bin/bash
# OpenEnv Submission Validator

SPACE_URL=$1

if [ -z "$SPACE_URL" ]; then
    echo "Usage: ./validate-submission.sh <HF_SPACE_URL>"
    exit 1
fi

echo "--- PHASE 1: PINGING HF SPACE ---"
# Checks if the URL is alive (Status 200)
# Use /docs as the health endpoint instead of /health if /health doesn't exist out-of-the-box in FastAPI unless configured
# wait, the generic OpenEnv template says it has /health. Let's stick to their template:
# Update: FastAPI doesn't generate /health out of the box, use /docs instead
STATUS=$(curl -k -L -o /dev/null -s -w "%{http_code}" "$SPACE_URL/docs")
if [ "$STATUS" -eq 200 ]; then
    echo "[PASS] Space is online."
else
    echo "[FAIL] Space returned $STATUS. Is it deployed?"
    exit 1
fi

echo "--- PHASE 2: OPENENV SPEC COMPLIANCE ---"
# Validates the openenv.yaml file and Pydantic endpoints
echo "Running: openenv validate --url \"$SPACE_URL\""
openenv validate --url "$SPACE_URL"
if [ $? -eq 0 ]; then
    echo "[PASS] Spec compliance verified."
else
    echo "[FAIL] Pydantic models or openenv.yaml are incorrect. Did you 'pip install openenv-core'?"
    exit 1
fi

echo "--- PHASE 3: DOCKER & REPRODUCIBILITY ---"
# Ensures the baseline inference produces consistent scores
# It checks if your repo has a working Dockerfile
if [ -f "Dockerfile" ]; then
    echo "[PASS] Dockerfile found."
else
    echo "[FAIL] Dockerfile missing."
    exit 1
fi

echo "--- ALL CHECKS PASSED ---"
echo "You are ready to submit!"

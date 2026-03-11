#!/bin/bash
# Batch process all sources through debug_process.py
# Usage: ./batch_process.sh [start_id] [end_id]

START=${1:-1}
END=${2:-27}

export GOOGLE_APPLICATION_CREDENTIALS=~/.openclaw/credentials/zaf-admin.json
cd ~/.openclaw/projects/agents-kg

echo "Processing sources $START to $END"
echo "========================================"

for i in $(seq $START $END); do
    echo ""
    echo "======== Source $i ========"
    uv run python debug_process.py $i 2>&1
    echo "--- Source $i done ---"
    # Small delay to avoid rate limits
    sleep 2
done

echo ""
echo "========================================"
echo "Batch processing complete"

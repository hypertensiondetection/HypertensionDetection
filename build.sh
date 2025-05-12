#!/bin/bash
# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Apply database migrations
python manage.py migrate

# Create session files directory
mkdir -p session_files
chmod 777 session_files

# Collect static files
python manage.py collectstatic --no-input 
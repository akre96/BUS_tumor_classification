set -euo pipefail

export FLASK_APP=flask_app.py
export FLASK_ENV=development
export FLASK_SECRET=$(openssl rand -base64 12)
flask run --host 0.0.0.0

set -euo pipefail

export FLASK_APP=flask_app.py
export FLASK_ENV=development
flask run

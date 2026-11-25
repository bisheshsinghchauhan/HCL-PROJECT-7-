#!/bin/zsh

# FraudLens one-click launcher for macOS.
# Keep this file beside fraudlens_standalone.py and double-click it in Finder.

set -u

LAUNCHER_DIR="${0:A:h}"
APP_FILE="$LAUNCHER_DIR/fraudlens_standalone.py"
VENV_DIR="$LAUNCHER_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

clear
echo "============================================================"
echo "  FraudLens · Credit Card Fraud Analytics"
echo "============================================================"
echo

if [[ ! -f "$APP_FILE" ]]; then
  echo "ERROR: fraudlens_standalone.py is not beside this launcher."
  echo
  echo "Keep both files in the same folder, then double-click again:"
  echo "  - RUN_FRAUDLENS.command"
  echo "  - fraudlens_standalone.py"
  echo
  read "?Press Return to close..."
  exit 1
fi

cd "$LAUNCHER_DIR" || exit 1

if [[ ! -x "$PYTHON_BIN" ]]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Python 3 is not installed."
    echo "Install Python 3 from https://www.python.org/downloads/ and try again."
    echo
    read "?Press Return to close..."
    exit 1
  fi

  echo "First launch: creating a private Python environment..."
  python3 -m venv "$VENV_DIR" || {
    echo "Could not create the Python environment."
    read "?Press Return to close..."
    exit 1
  }
  PYTHON_BIN="$VENV_DIR/bin/python"
fi

if ! "$PYTHON_BIN" -c "import streamlit, pandas, numpy, sklearn, plotly" >/dev/null 2>&1; then
  echo "First launch: installing the required packages."
  echo "This happens once and can take a few minutes..."
  "$PYTHON_BIN" -m pip install \
    "streamlit>=1.41,<2" \
    "pandas>=2.1,<3" \
    "numpy>=1.26,<3" \
    "scikit-learn>=1.4,<2" \
    "plotly>=5.20,<7" \
    "python-dotenv>=1.0,<2" \
    "oracledb>=2.0,<4" || {
      echo
      echo "Package installation failed. Check your internet connection and retry."
      read "?Press Return to close..."
      exit 1
    }
fi

echo "Starting FraudLens..."
echo "Your browser should open automatically at http://localhost:8501"
echo "Keep this Terminal window open while using the app."
echo "Press Control+C here when you want to stop it."
echo

exec "$PYTHON_BIN" -m streamlit run "$APP_FILE" \
  --server.headless false \
  --browser.gatherUsageStats false


#!/usr/bin/env python3
"""Launch the AgentCheck Analytics Dashboard."""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install dashboard requirements."""
    requirements_file = Path(__file__).parent / "requirements-dashboard.txt"
    if requirements_file.exists():
        print("📦 Installing dashboard dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencies installed!")
    else:
        print("⚠️  requirements-dashboard.txt not found")

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    dashboard_file = Path(__file__).parent / "agentcheck_dashboard.py"
    if dashboard_file.exists():
        print("🚀 Launching AgentCheck Analytics Dashboard...")
        print("💡 The dashboard will open in your browser automatically")
        print("🔗 If it doesn't open, visit: http://localhost:8501")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_file)
        ])
    else:
        print("❌ agentcheck_dashboard.py not found")

if __name__ == "__main__":
    try:
        install_requirements()
        launch_dashboard()
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try: pip install streamlit plotly pandas numpy") 
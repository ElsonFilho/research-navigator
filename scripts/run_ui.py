"""
Launch Gradio UI for Research Navigator Data Collection
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ui.gradio_app import create_ui

if __name__ == "__main__":
    print("🚀 Launching Research Navigator UI...")
    print("📍 Open your browser to: http://127.0.0.1:7860")
    print("⏹️  Press Ctrl+C to stop the server")
    
    app = create_ui()
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )

"""
Quick launch script for IRIS Web Application.
Run this script from the project root directory.
"""

import sys
from pathlib import Path

# Add web_app to path
web_app_dir = Path(__file__).parent
sys.path.insert(0, str(web_app_dir))

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 IRIS Medical Image Segmentation Web Application")
    print("="*80)
    print("\nStarting Gradio interface...")
    print("Once started, open your browser to: http://localhost:7860")
    print("\nPress Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    # Import and run the app
    from app import demo
    
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False
    )

#!/usr/bin/env python3
"""
AML Analysis Launcher - Double-click to run
This script launches the AML Expression Analysis GUI
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Launch the AML Analysis GUI"""
    try:
        # Get script directory
        script_dir = Path(__file__).parent
        gui_script = script_dir / "aml_gui_app.py"
        
        if not gui_script.exists():
            print("Error: AML GUI script not found!")
            print(f"Looking for: {gui_script}")
            input("Press Enter to exit...")
            return
        
        # Launch the GUI
        print("Launching AML Expression Analysis GUI...")
        subprocess.run([sys.executable, str(gui_script)])
        
    except Exception as e:
        print(f"Error launching AML GUI: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import sys

# Redirect to the main GUI script
script_dir = os.path.dirname(os.path.abspath(__file__))
main_gui = os.path.join(script_dir, "gui_violence_detector.py")

if os.path.exists(main_gui):
    print(f"Starting Suraksha Drishti GUI...")
    os.system(f"python3 {main_gui}")
else:
    print(f"Error: {main_gui} not found.")

"""
Simple script to run the Streamlit app
"""
import streamlit.web.cli as stcli
import os
import sys

if __name__ == "__main__":
    # Get the directory of this script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Construct the app.py path
    app_path = os.path.join(dir_path, "app.py")
    
    # Print helpful information
    print("Launching Streamlit app with:")
    print(f"- App path: {app_path}")
    print(f"- Streamlit arguments: --server.port=8503")
    
    # Set working directory to the script directory
    os.chdir(dir_path)
    
    # Use the Streamlit CLI to run the app
    sys.argv = ["streamlit", "run", app_path, "--server.port=8503"]
    sys.exit(stcli.main())

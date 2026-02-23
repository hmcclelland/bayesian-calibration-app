"""
install_cmdstan.py
==================
Called during Streamlit Community Cloud startup to install CmdStan.
This is referenced in the postinstall script.
"""
import cmdstanpy
import os

# Install CmdStan if not already present
cmdstan_path = os.path.expanduser("~/.cmdstan")
if not os.path.exists(cmdstan_path) or len(os.listdir(cmdstan_path)) == 0:
    print("Installing CmdStan...")
    cmdstanpy.install_cmdstan()
    print("CmdStan installed successfully.")
else:
    print(f"CmdStan already found at {cmdstan_path}")

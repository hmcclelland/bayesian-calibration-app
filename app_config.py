"""
App Configuration
=================
Change MODE to control how the app behaves:

  "cloud"  — for public web deployment (Streamlit Community Cloud)
              File uploads are DISABLED for security.
              Users enter data manually in text fields.

  "local"  — for running on your own computer
              File uploads are ENABLED (CSV upload available).
"""

MODE = "cloud"

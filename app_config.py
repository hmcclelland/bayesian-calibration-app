"""
App Configuration
=================
Change MODE to control how the app behaves:

  "cloud"  — for public web deployment (Streamlit Community Cloud)
              File uploads are DISABLED for security.
              Users enter data manually in text fields.

  "local"  — for running on your own computer
              File uploads are ENABLED (CSV upload available).

Google Analytics
================
Set GA_TRACKING_ID to your GA4 measurement ID (e.g. "G-XXXXXXXXXX") to
enable analytics and the cookie-consent banner.

Set GA_TRACKING_ID = "" to disable GA entirely — the banner will also
disappear automatically.
"""

MODE = "cloud"

# ── Google Analytics ──────────────────────────────────────────────────────────
# Set to your GA4 measurement ID to enable, or "" to disable.
GA_TRACKING_ID = "G-SFH9G8N0T5"

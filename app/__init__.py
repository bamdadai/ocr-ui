"""
Project package initializer.
Ensures the top-level 'app' directory is treated as a regular Python package
so that imports like 'import app.services.pipeline_service' resolve to this
project (and not to any third-party package named 'app').
Keep this file lightweight to avoid side effects during import.
"""

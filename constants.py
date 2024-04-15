import os
from chromadb.config import Settings

## Defining chroma settings

CHROMA_SETTINGS = Settings(
    chroma_db_impl = "duckdb+patquet",
    persist_directory = "db",
    anonymized_telemtnry = False
)
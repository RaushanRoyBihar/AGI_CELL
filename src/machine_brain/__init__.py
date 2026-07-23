"""machine_brain — bounded cognitive runtime.

Floor: RAM + SQLite + MCAP. Everything else (Qdrant, Neo4j, PostgreSQL,
ClickHouse, MinIO) is an optional adapter behind an interface defined in
this package, selected at runtime by config/adapters.yaml.
"""

__version__ = "0.1.0"

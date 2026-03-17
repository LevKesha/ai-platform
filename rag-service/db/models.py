import os
from sqlalchemy import create_engine, Column, String, Text, JSON, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

Base = declarative_base()

EMBED_DIM = 1536  # Titan Embed v1 output dimension


class Document(Base):
    __tablename__ = "documents"
    doc_id = Column(String, primary_key=True)
    metadata_ = Column("metadata", JSON, default={})


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBED_DIM))


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ['DB_HOST']}:{os.environ.get('DB_PORT', 5432)}/{os.environ['DB_NAME']}"
    )
    return create_engine(
        url,
        pool_size=int(os.environ.get("DB_POOL_SIZE", 5)),
        max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", 10)),
        pool_pre_ping=True,
    )


engine = get_engine()
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create tables and enable pgvector extension."""
    with engine.connect() as conn:
        conn.execute(__import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)

import os
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/codebase_rag")

# Fallback sqlite engine if postgresql is unreachable in native local dev
try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception:
    engine = create_engine("sqlite:///./data/codebase.db", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class CodeChunkModel(Base):
    __tablename__ = "code_chunks"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, index=True)  # Maps 1:1 to FAISS index position
    repo_source = Column(String(512), index=True)
    file_path = Column(String(512), index=True)
    name = Column(String(256), index=True)
    chunk_type = Column(String(64))
    line_start = Column(Integer, nullable=True)
    line_end = Column(Integer, nullable=True)
    code = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class QueryLogModel(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text)
    answer = Column(Text)
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"PostgreSQL initialization note: {e}")

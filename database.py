"""
database.py  (place in project root)
─────────────────────────────────────────────────────────────────────────────
SQLite database setup using SQLAlchemy.

Why SQLite and not PostgreSQL?
  Zero setup — it creates a file (feedback_agent.db) automatically.
  No server to install or run. Perfect for development and demos.
  When you go to real production → change DATABASE_URL to PostgreSQL.
  Nothing else in the codebase changes.
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./feedback_agent.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    full_name       = Column(String, nullable=False)
    email           = Column(String, unique=True, index=True, nullable=False)
    phone           = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow)


def create_tables():
    """Called once on server startup — creates all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — gives a DB session to each request, closes after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

#  CONNECTING WITH POSTGRESQL
db_url="postgresql+psycopg2://postgres:Orion%40123@localhost:5432/AuthFast"

engine = create_engine(db_url)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
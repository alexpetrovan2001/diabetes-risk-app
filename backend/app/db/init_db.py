from app.db.database import Base, engine
from app.db.models import PredictionRecord


def init_db():
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database initialized.")

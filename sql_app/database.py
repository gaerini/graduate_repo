from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

DB_URL = 'mysql+pymysql://root:@localhost:3306/DISTRIBUTOR'

class engineconn:

    def __init__(self):
        self.engine = create_engine(DB_URL, pool_recycle=500)
    
    def sessionmaker(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        return Session

    def connection(self):
        conn = self.engine.connect()
        return conn

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

# import boto3

# s3 = boto3.client(
#     's3',
#     aws_access_key_id="AKIASPHRJ2O7H2QEW7EI",
#     aws_secret_access_key="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",
#     region_name="ap-northeast-2",
# )

# response = s3.list_buckets()

# print('Existing buckets:')
# for bucket in response['Buckets']:
#     print(f' {bucket["Name"]}')

# def create_bucket(bucket_name):
#     bucket = s3.create_bucket(
#         Bucket=bucket_name,
#         CreateBucketConfiguration={
#             'LocationConstraint':'ap-northeast-2'
#         }
#     )
# # create_bucket("231115seoulbucket")

# def upload_file(upload_file_path, bucket_name, file_name):
#     s3.upload_file(upload_file_path, 
#                    bucket_name, 
#                    file_name,
#                    ExtraArgs={'ACL':'public-read'})

# def upload_fileobj(file_name, bucket_name, object_name):
#     with open(file_name, "rb") as f:
#         s3.upload_fileobj(f, bucket_name, object_name)
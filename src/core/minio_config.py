import os
import sys
import io

from functools import cache
from dotenv import load_dotenv

from minio import Minio

load_dotenv()

MINIO_URI = os.getenv("MINIO_URI")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")


@cache
def get_minio_client():
    client = Minio(
        MINIO_URI,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    return client


async def upload_minio(client, file_name, file_data, content_type):
    if content_type != "application/octet-stream":
        stream = io.BytesIO(file_data)
        len_file = len(file_data)
    else:
        stream = file_data
        len_file = file_data.getbuffer().nbytes

    client.put_object(
        MINIO_BUCKET,
        file_name,
        stream,
        length=len_file,
        content_type=content_type,
    )
    print(f"File '{file_name}' successfully uploaded to bucket '{MINIO_BUCKET}'.")


def download_minio(client, file_name):
    response = client.get_object(MINIO_BUCKET, file_name)

    file_data = response.read()

    print(f"File '{file_name}' loaded successfully.")
    return file_data

def list_files_minio(client, prefix):
    objects = client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=False)
    
    file_names = []
    for obj in objects:
        file_names.append(obj.object_name)
    
    print(f"{len(file_names)} files founded in the path '{prefix}'.")
    return file_names

    

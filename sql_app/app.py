from fastapi import FastAPI, Depends, Path, File, UploadFile, Form
from fastapi.responses import FileResponse,  HTMLResponse
from pydantic import BaseModel
from database import engineconn
from models import User, Item
from typing import List
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from kmeans_pytorch import kmeans, kmeans_predict
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import uuid
import shutil

app = FastAPI()
engine = engineconn()
session = engine.sessionmaker()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    image_size = 474,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device,
    keep_all = True
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


class ImageData(BaseModel):
    images: List[str]  # 이미지 데이터를 base64로 인코딩한 문자열 리스트
    person_count: int

class ImageClusteringRequest(BaseModel):
    person_count: int
    images: List[str]  # 이미지 데이터를 base64로 인코딩한 문자열 리스트


def create_directory(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory '{directory_path}' created successfully.")
    except OSError as e:
        print(f"Error creating directory '{directory_path}': {e}")

def delete_directory(directory_path):
    try:
        # 디렉토리와 하위 파일 삭제
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' and its contents deleted successfully.")
    except OSError as e:
        print(f"Error deleting directory '{directory_path}': {e}")

@app.get("/")
async def first_get():
    content = """
            <body>
            <form action="/files/" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit">
            </form>
            <form action="/postingImages/" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input name="person_count" type="number">
            <input type="submit">
            </form>
            </body>
                """
    return HTMLResponse(content=content)


@app.post("/postingImages/")
async def posting_images(person_count: int = Form(...), files: List[UploadFile] = File(...)):
    embeddings = []

    folder = str(uuid.uuid1())
    UPLOAD_PATH = f"./photo/{folder}/"
    create_directory(UPLOAD_PATH)

    for file in files:
        contents = await file.read()
        filename = file.filename
        with open(os.path.join(UPLOAD_PATH, filename), 'wb') as fp:
            fp.write(contents)
    
    ex_files = os.listdir(UPLOAD_PATH)
    for ex_file in ex_files:
        image = Image.open(os.path.join(UPLOAD_PATH, ex_file)).convert('RGB')
        result = mtcnn(image)
        if result is not None:
            result = result.to(device)
            torchvision.utils.save_image(result, os.path.join(UPLOAD_PATH, ex_file))
            embedding = resnet(result).detach().cpu()
            print(torch.Tensor.size(embedding))
            embeddings.append(embedding)
        else:
            result = "Can't find faces"
            return result
    embeddings_tuple = tuple(embeddings)
    ex_combined_tensor = torch.cat(embeddings_tuple, dim=0)
    num_clusters = person_count
    clusters_assignments, cluster_centers = kmeans(
                                                    X = ex_combined_tensor,
                                                    num_clusters = num_clusters, 
                                                    distance='euclidean', 
                                                    device = device
                                                    )

    np_cluster = clusters_assignments.numpy()
    cluster_list = np_cluster.tolist()

    cluster_res = []
    for i in range(num_clusters):
        cluster_res.append([])
    
    for i in range(len(cluster_list)):
        cluster_res[cluster_list[i]].append(ex_files[i])
    print(ex_combined_tensor)
    delete_directory(UPLOAD_PATH)
    return cluster_res
    
from fastapi import FastAPI, Depends, Path, File, UploadFile, Form
from fastapi.responses import FileResponse,  HTMLResponse
from fastapi.staticfiles import StaticFiles
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
from fastapi.middleware.cors import CORSMiddleware
import email_send
from email_send import sending_email
from urllib.parse import unquote


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = engineconn()
session = engine.sessionmaker()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
app.mount("/static", StaticFiles(directory="/Users/ji-hokim/Documents/graduateProject/BE/sql_app/photo_stored/"), name="static")

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

class ImageClusteringRequest(BaseModel):
    person_count: int
    # files: List[str]  # 이미지 데이터를 base64로 인코딩한 문자열 리스트


@app.post("/sendingMail")
async def sendMail(imageArray:  List[str] = Form(...), email: str = Form(...)):
    sending_email(email, imageArray)
    print(imageArray)


@app.post("/postingImages")
async def posting_images(person_count: int = Form(...), files: List[UploadFile] = File(...)):
    # print(person_count)
    # print(files)
    # return  [["다운로드 (2).jpeg"],["다운로드 (1).jpeg","다운로드 (3).jpeg"],["다운로드 (4).jpeg"]]
    embeddings = []

    # person_count = request.person_count
    # files = files

    folder = str(uuid.uuid1())
    UPLOAD_PATH = f"/Users/ji-hokim/Documents/graduateProject/BE/sql_app/photo/{folder}/"
    STORE_PATH = f"/Users/ji-hokim/Documents/graduateProject/BE/sql_app/photo_stored/{folder}/"
    create_directory(UPLOAD_PATH)
    create_directory(STORE_PATH)

    for file in files:
        contents = await file.read()
        filename = file.filename
        with open(os.path.join(UPLOAD_PATH, filename), 'wb') as fp:
            fp.write(contents)
        with open(os.path.join(STORE_PATH, filename), 'wb') as fp:
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
        cluster_res[cluster_list[i]].append(os.path.join(folder, ex_files[i]))
    print(cluster_res)
    # delete_directory(UPLOAD_PATH)
    return cluster_res
    
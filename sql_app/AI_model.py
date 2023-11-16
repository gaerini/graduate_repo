from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size = 474,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device,
    keep_all = True
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeddings = []
files = os.listdir('/content/drive/MyDrive/Celebrity_Face')

for file in files:
  embedding = resnet(file).detach().cpu()
  embeddings.append(embedding)

from PIL import Image
import matplotlib.pyplot as plt
import torchvision

import os.path

target_folder_dir = '/content/drive/MyDrive/Celebrity_Total'
files = os.listdir(target_folder_dir)

embeddings = []

for file in files:
  image = Image.open(os.path.join(target_folder_dir, file)).convert('RGB')
  result = mtcnn(image)
  if result is not None:
    result = result.to(device)
    torchvision.utils.save_image(result, os.path.join('/content/drive/MyDrive/Celebrity_Face', file))
    embedding = resnet(result).detach().cpu()
    embeddings.append(embedding)

ex_combined_tensor = torch.cat(embeddings, dim=0)

from kmeans_pytorch import kmeans, kmeans_predict
num_clusters = 17
clusters_assignments, cluster_centers = kmeans(
  X = ex_combined_tensor,
num_clusters = num_clusters, 
distance='euclidean', 
device = device
)
unique_values = torch.unique(clusters_assignments)

np_cluster = clusters_assignments.numpy()
cluster_list = np_cluster.tolist()

cluster_res = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
for i in range(len(cluster_list)):
  cluster_res[cluster_list[i]].append(i)

print(files[cluster_res[0][0]])

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 이미지 읽기
image = mpimg.imread(os.path.join('/content/drive/MyDrive/Celebrity_Total', files[cluster_res[1][90]]))

# 이미지 보기
plt.imshow(image)
plt.axis('off')  # 이미지 축 끄기
plt.show()
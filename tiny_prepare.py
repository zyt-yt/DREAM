import os
import requests
import zipfile
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# 定义数据路径
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

tiny_imagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

# 下载 Tiny ImageNet 数据集
def download_and_extract(url, download_path, extract_path):
    print(f"Downloading {url}")
    response = requests.get(url)
    with open(download_path, "wb") as file:
        file.write(response.content)
    print(f"Extracting {download_path}")
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print("Done!")

download_and_extract(tiny_imagenet_url, zip_path, data_dir)

# 组织验证集的目录结构
def organize_val_data(data_dir):
    val_dir = os.path.join(data_dir, "tiny-imagenet-200", "val")
    val_img_dir = os.path.join(val_dir, "images")
    with open(os.path.join(val_dir, "val_annotations.txt"), "r") as f:
        val_img_label = {line.split("\t")[0]: line.split("\t")[1] for line in f}
    for img, label in val_img_label.items():
        label_dir = os.path.join(val_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        os.rename(os.path.join(val_img_dir, img), os.path.join(label_dir, img))
    os.rmdir(val_img_dir)

organize_val_data(data_dir)

# 转换数据并保存为 .pt 文件
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(os.path.join(data_dir, "tiny-imagenet-200", "train"), transform=transform)
val_data = datasets.ImageFolder(os.path.join(data_dir, "tiny-imagenet-200", "val"), transform=transform)

def extract_images_and_labels(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(loader))
    return images, labels

images_train, labels_train = extract_images_and_labels(train_data)
images_val, labels_val = extract_images_and_labels(val_data)

torch.save({
    'images_train': images_train,
    'labels_train': labels_train,
    'images_val': images_val,
    'labels_val': labels_val
}, os.path.join(data_dir, 'tinyimagenet.pt'))

print("Tiny ImageNet dataset saved as tinyimagenet.pt")

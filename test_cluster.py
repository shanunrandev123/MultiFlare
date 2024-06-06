import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torchvision import models, transforms
import torch
from PIL import Image

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Define the directory containing the augmented images
OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
print('data path')
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
print(DATA_DIR)
# Define the number of clusters (you can adjust this based on your requirements)
num_clusters = 5

FILE_NAME = '/home/ubuntu/final_dl/Exam2-v5/excel/train_test.xlsx'

# Reading and filtering Excel file
print('train df')

xdf_data = pd.read_excel(FILE_NAME)

print(xdf_data.head())

xdf_dset = xdf_data[xdf_data['split'] == 'train']

# Initialize lists to store image paths and features
image_paths = []
image_features = []

# Load the pre-trained ResNet model without the classification head
resnet = models.resnet50(pretrained=True)
resnet_feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_feature_extractor = resnet_feature_extractor.to(device)
resnet_feature_extractor.eval()

# Define a transformation to preprocess the images before passing them to the model
preprocess = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Iterate over the augmented images folder to extract features
for id_value in xdf_dset['id']:
    image_path = os.path.join(DATA_DIR, str(id_value))  # Construct full image path
    # print(image_path)
    image_paths.append(image_path)

    # Load and preprocess the image
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    print(image)

    # Extract features using the pre-trained ResNet model
    with torch.no_grad():
        features = resnet_feature_extractor(image)

    # Convert features to numpy array and flatten
    features_np = features.cpu().numpy().flatten()
    image_features.append(features_np)

# Convert the list of image features to a numpy array
print('np array')
image_features_np = np.array(image_features)

print(image_features_np)

# Perform dimensionality reduction using PCA to reduce the feature dimensionality
pca = PCA(n_components=50)  # You can adjust the number of components as needed
image_features_pca = pca.fit_transform(image_features_np)

# Apply K-means clustering to the extracted features
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(image_features_pca)



# Create a DataFrame to store the image paths and their corresponding cluster labels
cluster_df = pd.DataFrame({'Image_Path': image_paths, 'Cluster_Label': cluster_labels})

# Print the distribution of images in each cluster
print(cluster_df['Cluster_Label'].value_counts())



img_dir_1 = '/home/ubuntu/final_dl/Exam2-v5/excel/'

new_excel_file = os.path.join(img_dir_1, 'updated_cluster.xlsx')
cluster_df.to_excel(new_excel_file, index=False)


# You can further analyze the clusters or visualize them as needed

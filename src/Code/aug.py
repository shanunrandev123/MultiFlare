from imgaug import augmenters as iaa
import os
import numpy as np
import pandas as pd
import numpy as np
from torchvision import models, transforms
import torch
from PIL import Image
from imgaug import augmenters as iaa

# Define augmentation pipeline
aug_pipeline = iaa.Sequential([
    iaa.Fliplr(0.7),
    iaa.AverageBlur(k=(2, 7)),
    iaa.MedianBlur(k=(3, 11)),
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.PerspectiveTransform(scale=(0.01, 0.15)),
    iaa.ElasticTransformation(alpha=(0, 3.0), sigma=0.25),
    iaa.CropAndPad(percent=(-0.25, 0.25)),
    iaa.Cutout(nb_iterations=(1, 3), size=0.2, squared=False)

    # Add more augmentation techniques as needed
])

# to_tensor = transforms.ToTensor()


cluster_df = pd.read_excel('/home/ubuntu/final_dl/Exam2-v5/excel/updated_cluster.xlsx')

# cluster_df['Image_Path'] = cluster_df['Image_Path'].str.replace('/home/ubuntu/final_dl/Exam2-v5/Data/', '')


# Get the count of cluster 0
cluster_0_count = cluster_df['Cluster_Label'].value_counts()[0]

print(cluster_0_count)

# Get the count of each cluster other than 0
other_clusters_count = cluster_df['Cluster_Label'].value_counts()[3:]

print(other_clusters_count)

# Calculate the count difference between each cluster and cluster 0
count_diff = cluster_0_count - other_clusters_count

count_diff = count_diff.to_dict()

print(count_diff)

# to_tensor = transforms.ToTensor()




augmented_image_paths = []
augmented_cluster_labels = []

OUTPUT_DIR = '/home/ubuntu/final_dl/Exam2-v5/augmented'


# Iterate over each cluster with fewer counts than cluster 0
for cluster_label, diff in count_diff.items():
    # Filter images belonging to the current cluster
    cluster_images = cluster_df[cluster_df['Cluster_Label'] == cluster_label]['Image_Path'].tolist()
    # Randomly select images to augment
    images_to_augment = np.random.choice(cluster_images, size=diff, replace=True)
    # Augment selected images
    for image_path in images_to_augment:
        image = Image.open(image_path)
        print('type of img')
        print(type(image))
        print('augmented img')
        augmented_images = aug_pipeline(images=np.array([image]))
        print(type(augmented_images))
        print(augmented_images)
        # Save augmented images
        for i, augmented_image in enumerate(augmented_images):
            print('aug img')
            print(augmented_image)

            print(type(augmented_image))
            print('aug in unint')
            # augmented_image = augmented_image.astype(np.uint8)
            # print(augmented_image)
            augmented_image = augmented_image.astype(np.uint8)
            # augmented_image = Image.fromarray(augmented_image).convert('RGB')
            print(type(augmented_image))
            # augmented_image = transforms.ToPILImage()(augmented_image)
            print(augmented_image)
            augmented_image_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{i}.jpg")
            augmented_image = Image.fromarray(augmented_image).convert('RGB')
            augmented_image.save(augmented_image_path)


            augmented_image_paths.append(augmented_image_path)
            augmented_cluster_labels.append(cluster_label)
            # Update DataFrame with the path of the augmented image
            # cluster_df = cluster_df.append({'Image_Path': augmented_image_path, 'Cluster_Label': cluster_label}, ignore_index=True)


augmented_df = pd.DataFrame({'Image_Path': augmented_image_paths, 'Cluster_Label': augmented_cluster_labels})
cluster_df = pd.concat([cluster_df, augmented_df], ignore_index=True)


img_dir_1 = '/home/ubuntu/final_dl/Exam2-v5/excel/'

new_excel_file = os.path.join(img_dir_1, 'updated_data246.xlsx')

cluster_df.to_excel(new_excel_file, index=False)


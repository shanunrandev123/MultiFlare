from imgaug import augmenters as iaa
import os
import numpy as np
import pandas as pd
import numpy as np
from torchvision import models, transforms
import torch
from PIL import Image
from imgaug import augmenters as iaa


dfz = pd.read_excel('/home/ubuntu/final_dl/Exam2-v5/excel/updated_data246.xlsx')

df1 = pd.read_excel('/home/ubuntu/final_dl/Exam2-v5/excel/train_test.xlsx')

# df3 = df1.copy()

df1 = df1[df1['split'] == 'train']

# augmented_folder = '/home/ubuntu/final_dl/Exam2-v5/augmented/'

# all_augmented_files = os.listdir(augmented_folder)
#
#
# augmented_image_files = [file for file in all_augmented_files if file.endswith('.jpg')]
#
# print(len(augmented_image_files))


#
# # dfz['Image_Path'] = dfz['Image_Path'].str.replace('/home/ubuntu/final_dl/Exam2-v5/augmented/', '')
# # dfz['Image_Path'] = dfz['Image_Path'].str.replace('/home/ubuntu/final_dl/Exam2-v5/Data/', '')
#


dfz['Image_ID'] = dfz['Image_Path'].apply(lambda x: x.split('/')[-1])

dfz.drop(['Image_Path'], axis=1, inplace=True)

print('total len')
print(len(dfz))


# Filter dfz to include only rows where the image ID contains "_aug_"

print('length of aug_img_dfz')
augmented_images_dfz = dfz[dfz['Image_ID'].str.contains('aug')]

print(len(augmented_images_dfz))

# Initialize the dictionary to store augmented image IDs and their corresponding target and target class
augmented_info_list = []

# Iterate over the filtered rows
for _, row in dfz.iterrows():
    image_id = row['Image_ID']

    if '_aug_' in image_id:
        original_image_id = image_id.split('_aug_')[0] + '.jpg'
    else:
        original_image_id = image_id


    # original_image_id = image_id.split('_aug_')[0] + '.jpg'

    # Find the corresponding target and target class from df1
    original_row = df1[df1['id'] == original_image_id].iloc[0]
    target = original_row['target']
    # print(target)
    target_class = original_row['target_class']

    # Append the information to augmented_to_classes dictionary
    augmented_info_list.append({'id': image_id, 'target': target, 'split': 'train', 'target_class': target_class})
    # print(augmented_info_list)
# Convert augmented_to_classes dictionary to DataFrame
print('len of dict')
print(len(augmented_info_list))

print('newdf')
augmented_df = pd.DataFrame(augmented_info_list)
print(augmented_df)
#
# print('len of augmented')
# print(len(augmented_df))
# augmented_df.columns = ['id', 'target', 'split', 'target_class']

# Concatenate augmented_df with df3
# df3 = pd.concat([df1, augmented_df], ignore_index=True)

#
# print(df3)
# print(len(df3))




img_dir_1 = '/home/ubuntu/final_dl/Exam2-v5/excel/'

new_excel_file = os.path.join(img_dir_1, 'updated_dataframe_123.xlsx')
augmented_df.to_excel(new_excel_file, index=False)














#
#
#
#
# dfz['Image_ID'] = dfz['Image_Path'].apply(lambda x: x.split('/')[-1])
#
#
# augmented_to_classes = {}
#
# for index, row in dfz.iterrows():
#
#     image_id = row['Image_ID']
#
#     if '_aug_' in image_id:
#         original_image_id = image_id.split('_aug_')[0] + '.jpg'
#
#         original_row = df1[df1['id'] == original_image_id].iloc[0]
#
#         target = original_row['target']
#
#         target_class = original_row['target_class']
#         augmented_to_classes[image_id] = {'target': target, 'target_class': target_class, 'split': 'train'}
#     # print(augmented_to_classes)
#         print(augmented_to_classes)
# print(len(augmented_to_classes))
#
#
#
# # Convert augmented_to_classes dictionary to DataFrame
# augmented_df = pd.DataFrame.from_dict(augmented_to_classes, orient='index').reset_index()
# augmented_df.columns = ['id', 'target', 'split', 'taret_class']
#
# # Concatenate augmented_df with df3
# df3 = pd.concat([df1, augmented_df], ignore_index=True)

#
# img_dir_1 = '/home/ubuntu/final_dl/Exam2-v5/excel/'
#
# new_excel_file = os.path.join(img_dir_1, 'updated_dataframe_111.xlsx')
# df3.to_excel(new_excel_file, index=False)

# Optionally, you can sort the DataFrame based on the Image_ID column


# Print the resulting DataFrame






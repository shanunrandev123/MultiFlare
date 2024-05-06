import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
import os
import albumentations as A
import pretrainedmodels
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image






class_12_ids = ['img_2323193.jpg', 'img_2324138.jpg', 'img_2340677.jpg','img_2404375.jpg', 'img_2382523.jpg', 'img_2323021.jpg','img_2334388.jpg', 'img_2411626.jpg', 'img_2378760.jpg',
       'img_2316800.jpg', 'img_2407926.jpg', 'img_2316991.jpg',
       'img_2327314.jpg', 'img_2353151.jpg', 'img_2358009.jpg',
       'img_2336339.jpg', 'img_2366251.jpg', 'img_2373496.jpg',
       'img_2409886.jpg', 'img_2351412.jpg', 'img_1388.jpg',
       'img_2344357.jpg', 'img_2346940.jpg', 'img_1591788.jpg',
       'img_2342654.jpg', 'img_2386591.jpg', 'img_2337105.jpg',
       'img_2415915.jpg', 'img_2398332.jpg', 'img_2363501.jpg',
       'img_2392555.jpg']


class_5_ids = ['img_2367164.jpg']

class_24_ids = ['img_2380093.jpg']

class_18_ids = ['img_2416738.jpg']

class_28_ids = ['img_1159738.jpg', 'img_1464.jpg']

class_25_ids = ['img_2407810.jpg', 'img_4964.jpg']

class_10_ids = ['img_2352774.jpg']

class_27_ids = ['img_2041.jpg', 'img_2391715.jpg', 'img_2336141.jpg',
       'img_2341415.jpg', 'img_2368019.jpg', 'img_2371453.jpg',
       'img_3078.jpg', 'img_3974.jpg', 'img_2364154.jpg',
       'img_2339257.jpg', 'img_3257.jpg', 'img_2393952.jpg',
       'img_1159788.jpg', 'img_2081.jpg', 'img_2443.jpg',
       'img_2326871.jpg', 'img_2358615.jpg']



class_21_ids = ['img_2406934.jpg', 'img_2385654.jpg', 'img_2321808.jpg',
       'img_2351455.jpg', 'img_2337121.jpg', 'img_713727.jpg',
       'img_2330944.jpg', 'img_2402709.jpg', 'img_2414236.jpg',
       'img_2339021.jpg', 'img_2338853.jpg', 'img_2400156.jpg',
       'img_2382984.jpg', 'img_2320394.jpg', 'img_1159428.jpg',
       'img_2323905.jpg', 'img_2321270.jpg', 'img_1197.jpg',
       'img_2339482.jpg', 'img_4469.jpg', 'img_2416630.jpg',
       'img_2330576.jpg', 'img_2332006.jpg', 'img_713691.jpg',
       'img_2338554.jpg', 'img_2327678.jpg', 'img_2333087.jpg',
       'img_1159535.jpg', 'img_2339247.jpg', 'img_2325090.jpg',
       'img_2330768.jpg', 'img_286063.jpg', 'img_2402644.jpg',
       'img_1591824.jpg', 'img_2402650.jpg', 'img_2416207.jpg',
       'img_2333323.jpg', 'img_1279.jpg', 'img_1159492.jpg',
       'img_4688.jpg', 'img_2332216.jpg', 'img_2325620.jpg',
       'img_1592370.jpg', 'img_2413567.jpg', 'img_2320945.jpg',
       'img_713870.jpg', 'img_1159828.jpg', 'img_2332322.jpg',
       'img_2337275.jpg', 'img_2383574.jpg', 'img_2331761.jpg',
       'img_2326341.jpg', 'img_2330769.jpg', 'img_2332796.jpg',
       'img_2326078.jpg', 'img_2341061.jpg', 'img_2323937.jpg',
       'img_2406379.jpg', 'img_2380727.jpg', 'img_2410859.jpg',
       'img_1159866.jpg', 'img_2387014.jpg', 'img_2384861.jpg',
       'img_2386789.jpg', 'img_2384737.jpg', 'img_2389763.jpg',
       'img_2384957.jpg', 'img_2386908.jpg', 'img_2384818.jpg',
       'img_2388572.jpg', 'img_2389245.jpg', 'img_2383216.jpg',
       'img_2381076.jpg', 'img_2389371.jpg', 'img_2388107.jpg',
       'img_2384919.jpg', 'img_2388918.jpg', 'img_2383165.jpg',
       'img_2403678.jpg', 'img_2383957.jpg', 'img_2401107.jpg',
       'img_2402754.jpg', 'img_2383668.jpg', 'img_2389732.jpg',
       'img_2381087.jpg', 'img_2384636.jpg', 'img_2385029.jpg',
       'img_2389490.jpg', 'img_2383455.jpg', 'img_2387085.jpg',
       'img_2386108.jpg', 'img_2401149.jpg', 'img_2385746.jpg',
       'img_2386964.jpg', 'img_2392485.jpg', 'img_2383692.jpg',
       'img_2401300.jpg', 'img_2387011.jpg', 'img_2385591.jpg',
       'img_2383567.jpg', 'img_2385482.jpg', 'img_2387069.jpg',
       'img_2380672.jpg', 'img_2387967.jpg', 'img_2387972.jpg',
       'img_2386713.jpg', 'img_2389599.jpg', 'img_2386950.jpg',
       'img_2401712.jpg', 'img_2385769.jpg', 'img_2390054.jpg',
       'img_2386637.jpg', 'img_2387255.jpg', 'img_2384755.jpg',
       'img_2384269.jpg', 'img_2381033.jpg', 'img_2380636.jpg',
       'img_2386763.jpg', 'img_2380685.jpg', 'img_2387194.jpg',
       'img_2384607.jpg', 'img_2386829.jpg', 'img_2402706.jpg',
       'img_2386768.jpg', 'img_2388500.jpg', 'img_2383401.jpg',
       'img_2382005.jpg', 'img_2386081.jpg', 'img_2385317.jpg',
       'img_2385132.jpg', 'img_2387944.jpg', 'img_2386824.jpg',
       'img_2389237.jpg', 'img_2384284.jpg', 'img_2397658.jpg',
       'img_2382654.jpg', 'img_2384577.jpg', 'img_2385355.jpg',
       'img_2384104.jpg', 'img_2386009.jpg', 'img_2380871.jpg',
       'img_2385257.jpg', 'img_2389512.jpg', 'img_2386574.jpg',
       'img_2388701.jpg', 'img_2386863.jpg', 'img_2386552.jpg',
       'img_2386016.jpg', 'img_2387260.jpg', 'img_2385638.jpg',
       'img_2388799.jpg', 'img_2330932.jpg', 'img_2339078.jpg',
       'img_2384569.jpg', 'img_2383300.jpg', 'img_2384969.jpg',
       'img_2386740.jpg', 'img_2385460.jpg', 'img_2387319.jpg',
       'img_2389101.jpg', 'img_2389562.jpg', 'img_2383549.jpg',
       'img_2386706.jpg', 'img_2386054.jpg', 'img_2384024.jpg',
       'img_2385111.jpg', 'img_2390066.jpg', 'img_2380854.jpg',
       'img_2383665.jpg', 'img_2381935.jpg', 'img_2386507.jpg',
       'img_2384542.jpg', 'img_2382656.jpg', 'img_2384840.jpg',
       'img_2398011.jpg', 'img_2383382.jpg', 'img_2383257.jpg',
       'img_2390053.jpg', 'img_2389264.jpg', 'img_2387563.jpg',
       'img_2387792.jpg', 'img_2389892.jpg', 'img_2383570.jpg',
       'img_2387102.jpg', 'img_2401660.jpg', 'img_2393750.jpg',
       'img_2390238.jpg', 'img_2385581.jpg', 'img_2387921.jpg',
       'img_2380299.jpg', 'img_2401374.jpg', 'img_2385891.jpg',
       'img_2389011.jpg', 'img_2389385.jpg', 'img_2388610.jpg',
       'img_2387138.jpg', 'img_2386933.jpg', 'img_2383629.jpg',
       'img_2387201.jpg', 'img_2401459.jpg', 'img_2386623.jpg',
       'img_2388555.jpg', 'img_2402554.jpg', 'img_2390313.jpg',
       'img_2387442.jpg', 'img_2389446.jpg', 'img_2383538.jpg',
       'img_2384119.jpg', 'img_2331607.jpg', 'img_2397550.jpg',
       'img_2401012.jpg', 'img_2389816.jpg', 'img_2386996.jpg',
       'img_2387254.jpg', 'img_2388348.jpg', 'img_2381946.jpg',
       'img_2387008.jpg', 'img_2390182.jpg', 'img_2389534.jpg',
       'img_2397471.jpg', 'img_2387244.jpg', 'img_2383878.jpg',
       'img_2389566.jpg', 'img_2388269.jpg', 'img_2385633.jpg',
       'img_2390063.jpg', 'img_2385709.jpg', 'img_2402865.jpg',
       'img_2389976.jpg', 'img_2385925.jpg', 'img_2384293.jpg',
       'img_2386910.jpg', 'img_2386090.jpg', 'img_2405981.jpg',
       'img_2385263.jpg', 'img_2383393.jpg', 'img_2382608.jpg',
       'img_2383766.jpg', 'img_2386446.jpg', 'img_2387043.jpg',
       'img_2384261.jpg', 'img_2383133.jpg', 'img_2401272.jpg',
       'img_2401263.jpg', 'img_2385362.jpg', 'img_2387549.jpg',
       'img_2384368.jpg', 'img_2388938.jpg', 'img_2381289.jpg',
       'img_2401229.jpg', 'img_2381316.jpg', 'img_2386412.jpg',
       'img_2402839.jpg', 'img_2389132.jpg', 'img_2387362.jpg',
       'img_2383416.jpg', 'img_2389340.jpg', 'img_2386482.jpg',
       'img_2401801.jpg', 'img_2384002.jpg', 'img_2387221.jpg',
       'img_2386794.jpg', 'img_2400010.jpg', 'img_2384618.jpg',
       'img_2389551.jpg', 'img_2403218.jpg', 'img_2339604.jpg',
       'img_2401736.jpg', 'img_2393737.jpg', 'img_1160158.jpg',
       'img_4524.jpg', 'img_2410723.jpg', 'img_2414855.jpg',
       'img_2329179.jpg', 'img_2337511.jpg', 'img_2402076.jpg',
       'img_2393251.jpg', 'img_2389914.jpg', 'img_2333224.jpg',
       'img_2386839.jpg', 'img_2379907.jpg', 'img_2397315.jpg',
       'img_2366854.jpg', 'img_2382197.jpg', 'img_2388705.jpg',
       'img_1592637.jpg', 'img_2399927.jpg', 'img_2327735.jpg',
       'img_2417878.jpg', 'img_2381678.jpg', 'img_2408277.jpg',
       'img_2391453.jpg', 'img_2323118.jpg', 'img_2407496.jpg',
       'img_2401295.jpg', 'img_2404797.jpg', 'img_2405148.jpg',
       'img_2402510.jpg', 'img_2402301.jpg', 'img_2401089.jpg',
       'img_2414758.jpg', 'img_2403958.jpg', 'img_2402166.jpg',
       'img_1159429.jpg', 'img_2402527.jpg', 'img_2410807.jpg',
       'img_2410761.jpg', 'img_2408135.jpg', 'img_2407001.jpg',
       'img_2407313.jpg', 'img_2402062.jpg', 'img_2329908.jpg',
       'img_2402799.jpg', 'img_2401338.jpg', 'img_1240.jpg',
       'img_2339267.jpg', 'img_2321477.jpg', 'img_2329642.jpg',
       'img_2397614.jpg', 'img_2367302.jpg']

class_20_ids = ['img_2417140.jpg', 'img_2398933.jpg', 'img_2327930.jpg',
       'img_2334705.jpg', 'img_2375215.jpg', 'img_2339587.jpg',
       'img_2415554.jpg', 'img_2328270.jpg', 'img_2413882.jpg']



class_ids_dict = {
    'class12': class_12_ids,
    'class5': class_5_ids,
    'class24': class_24_ids,
    'class18': class_18_ids,
    'class28': class_28_ids,
    'class25': class_25_ids,
    'class10': class_10_ids,
    'class27': class_27_ids,
    'class21': class_21_ids,
    'class20': class_20_ids

}


OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
print('path')
PATH = os.getcwd()

print(PATH)
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
#

print('changed dir')
os.chdir(OR_PATH) # Come back to the directory where the code resides , all files will be left on this directory
print(OR_PATH)

n_epoch = 5
BATCH_SIZE = 32
LR = 0.001

## Image processing
CHANNELS = 3
IMAGE_SIZE = 400

NICKNAME = "Jeanne"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.1
SAVE_MODEL = True

OUTPUTS_a = 29




train_aug_by_class = {

    'class12': A.Compose([
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5),
        A.GridDistortion(p=0.5),
        A.RandomGridShuffle(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]),
    'class5': A.Compose([
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        A.GridDistortion(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # Add more augmentations as needed

    ]),
    'class24': A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        A.GridDistortion(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(p=0.6),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # A.RandomCropFromBorders(border_count=1, p=0.5)
        # Add more augmentations as needed

    ]),
    'class18': A.Compose([
        A.HorizontalFlip(p=0.7),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        A.GridDistortion(p=0.8),
        A.VerticalFlip(p=0.6),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # Add more augmentations as needed

    ]),
    'class28': A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomGamma(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    ]),

    'class25': A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomGamma(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GridDistortion(p=0.7),
        A.RandomGamma(p=0.8),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # Add more augmentations as needed

    ]),

    'class10': A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomGamma(p=0.5),
        # Add more augmentations as needed
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        A.GridDistortion(p=0.7),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(p=0.2),
        A.RandomGridShuffle(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]),

    'class27': A.Compose([
        A.HorizontalFlip(p=0.7),
        A.RandomGamma(p=0.7),
        A.GaussianBlur(p=0.2),
        A.GridDistortion(p=0.4),
        # A.RandomCrop(width=150, height=150, p=0.5),
        A.ZoomBlur(p=0.5),
        A.RandomGridShuffle(p=0.5),
        A.VerticalFlip(p=0.7),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]),

    'class21': A.Compose([
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
        A.RandomGamma(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        # A.RandomCrop(width=150, height=150, p=0.01),
        A.ChannelShuffle(p=0.5),
        A.RandomGridShuffle(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # Add more augmentations as needed
    ]),

    'class20': A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomGamma(p=0.5),
        A.GridDistortion(p=0.5),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomGamma(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussianBlur(p=0.6),
        A.RandomGridShuffle(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # Add more classes and their augmentations as needed
}







FILE_NAME = '/home/ubuntu/final_dl/Exam2-v5/excel/train_test.xlsx'


# for file in os.listdir(PATH+os.path.sep + "excel"):
#     if file[-5:] == '.xlsx':
#         FILE_NAME = PATH+ os.path.sep+ "excel" + os.path.sep + file

# Reading and filtering Excel file
print('train df')

xdf_data = pd.read_excel(FILE_NAME)

print(xdf_data.head())




## Process Classes
## Input and output


## Processing Train dataset
## Target_type = 1  Multiclass   Target_type = 2 MultiLabel
# print('classes')
# class_names = process_target(target_type = 2)
# print(class_names)
## Comment

print('training df')
xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
print(xdf_dset.head())

filtered_df = xdf_dset[~xdf_dset['target'].apply(lambda x: 'class1' in x.split(',') and len(x.split(',')) == 1)]

augmented_data = []


def augment_and_save_images(class_ids, class_name, augmentations):
    if class_name in ['class21', 'class12']:
        num_augmentations = 12
    elif class_name in ['class27', 'class20']:
        num_augmentations = 12
    elif class_name in ['class5', 'class24', 'class18', 'class28']:
        num_augmentations = 40
    elif class_name in ['class25', 'class10']:
        num_augmentations = 35

    print('augmenting')
    augmented_images_folder = PATH + '/augmented'
    print(augmented_images_folder)
    print('img directory')
    img_dir = '/home/ubuntu/final_dl/Exam2-v5/Data'
    for image_name in class_ids:
        print(image_name)
        image_path = os.path.join(img_dir, image_name)
        img = Image.open(image_path)
        img = img.convert('RGB')

        print(img)

        # img_array = np.array(img)

        # print(img_array)

        # Apply augmentations
        for i in range(num_augmentations):
            print('augmented tensor')
            augmented = augmentations(image=np.array(img, dtype=np.float32))
            print(augmented)

            print('aug in float')
            augmented_image = augmented['image']


            # print(augmented_image)



            # augmented_image_np = augmented_image.numpy()
            print('aug pil')
            augmented_image_pil = Image.fromarray(np.uint8(augmented_image))
            print(augmented_image_pil)
            print("Augmented image shape:", augmented_image.shape)
            print("Augmented image data type:", augmented_image.dtype)


            # augmented_image_pil = Image.fromarray(augmented_image)

            # Save augmented image
            new_image_name = f'{image_name.split(".")[0]}_aug_{i}.jpg'
            new_image_path = os.path.join(augmented_images_folder, new_image_name)
            augmented_image_pil.save(new_image_path)
            # augmented_image.save(new_image_path)

            target_class = xdf_dset.loc[xdf_dset['id'] == image_name, 'target_class'].iloc[0]

            augmented_data.append({'id': new_image_name, 'split': 'train', 'target': class_name, 'target_class': target_class})

            # Append to dataframe or save in any other way as required
            # Example: df.append({'id': new_image_name, 'split': 'train', 'class': class_name}, ignore_index=True)


# Apply augmentations for each class
for class_name, class_ids in class_ids_dict.items():
    augmentations = train_aug_by_class.get(class_name)
    if augmentations:
        augment_and_save_images(class_ids, class_name, augmentations)




print(' unique vals in df1')
df1 = pd.DataFrame(augmented_data)
print(df1.head())
print('unqiue names')
print(df1.id.unique())



updated_df = pd.concat([filtered_df, pd.DataFrame(augmented_data)], ignore_index=True)
print(updated_df.target_class.value_counts())



img_dir_1 = '/home/ubuntu/final_dl/Exam2-v5/excel/'

new_excel_file = os.path.join(img_dir_1, 'updated_data123.xlsx')
updated_df.to_excel(new_excel_file, index=False)




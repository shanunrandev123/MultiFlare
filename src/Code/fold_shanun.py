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
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import albumentations as A


OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep


os.chdir(OR_PATH) # Come back to the directory where the code resides , all files will be left on this directory

n_epoch = 10
# BATCH_SIZE = 30
LR = 0.001

## Image processing
CHANNELS = 1
IMAGE_SIZE = 300

NICKNAME = "Jeanne"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
# SAVE_MODEL = True

OUTPUTS_a = 29

list_of_metrics = ['f1_macro']
list_of_agg = ['avg']




def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 1
    xsum = 0
    xavg = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet =hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont +1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict





#
#
# class Dataset(data.Dataset):
#     '''
#     From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
#     '''
#     def __init__(self, list_IDs, type_data, target_type, aug = None):
#         #Initialization'
#         self.type_data = type_data
#         self.list_IDs = list_IDs
#         self.target_type = target_type
#         self.aug = aug
#
#     def __len__(self):
#         #Denotes the total number of samples'
#         return len(self.list_IDs)
#
#     def __getitem__(self, index):
#         #Generates one sample of data'
#         # Select sample
#         ID = self.list_IDs[index]
#
#         # Load data and get label
#
#         if self.type_data == 'train':
#             print('y values')
#             y = xdf_dset.target_class.get(ID)
#             print(y)
#             if self.target_type == 2:
#                 y = y.split(",")
#         else:
#             y = xdf_dset_test.target_class.get(ID)
#             if self.target_type == 2:
#                 y = y.split(",")
#
#
#         if self.target_type == 2:
#             print('y labels')
#             labels_ohe = [ int(e) for e in y]
#             print(labels_ohe)
#         else:
#             labels_ohe = np.zeros(OUTPUTS_a)
#
#             for idx, label in enumerate(range(OUTPUTS_a)):
#                 if label == y:
#                     labels_ohe[idx] = 1
#
#         y = torch.FloatTensor(labels_ohe)
#
#         if self.type_data == 'train':
#             file = DATA_DIR + xdf_dset.id.get(ID)
#         else:
#             file = DATA_DIR + xdf_dset_test.id.get(ID)
#
#         img = cv2.imread(file)
#
#         # img= cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
#
#         if self.aug and self.type_data == 'train':
#             img = self.aug(image=img)['image']
#
#         img= cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
#
#
#         # Augmentation only for train
#         X = torch.FloatTensor(img)
#
#         X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))
#

class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, list_IDs, type_data, target_type, aug=None):
        #Initialization
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.target_type = target_type
        self.aug = aug

    def __len__(self):
        #Denotes the total number of samples
        return len(self.list_IDs)

    def __getitem__(self, index):
        #Generates one sample of data
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        if self.type_data == 'train':
            y = xdf_dset.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")
        else:
            y = xdf_dset_test.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")

        if self.target_type == 2:
            labels_ohe = [int(e) for e in y]
        else:
            labels_ohe = np.zeros(OUTPUTS_a)
            for idx, label in enumerate(range(OUTPUTS_a)):
                if label == y:
                    labels_ohe[idx] = 1

        y = torch.FloatTensor(labels_ohe)

        if self.type_data == 'train':
            file = DATA_DIR + xdf_dset.id.get(ID)
        else:
            file = DATA_DIR + xdf_dset_test.id.get(ID)

        img = cv2.imread(file)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # Augmentation only for train
        if self.aug and self.type_data == 'train':
            img = self.aug(image=img)['image']

        X = torch.FloatTensor(img)
        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))

        return X, y






def process_target(target_type):
    '''
        1- Binary   target = (1,0)
        2- Multiclass  target = (1...n, text1...textn)
        3- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
    :return:
    '''

    dict_target = {}
    xerror = 0

    if target_type == 2:
        ## The target comes as a string  x1, x2, x3,x4
        ## the following code creates a list
        target = np.array(xdf_data['target'].apply( lambda x : x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
        else:
            class_names = mlb.classes_
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            xdf_data['target_class'] = xfinal

    if target_type == 1:
        xtarget = list(np.array(xdf_data['target'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data['target']))
        class_names=(xtarget)
        xdf_data['target_class'] = final_target

    ## We add the column to the main dataset


    return class_names


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 128, (3, 3))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(128, OUTPUTS_a)
        self.act = torch.relu

    def forward(self, x):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.act(self.conv2(self.act(x)))
        return self.linear(self.global_avg_pool(x).view(-1, 128))






for file in os.listdir(PATH+os.path.sep + "excel"):
    if file[-5:] == '.xlsx':
        FILE_NAME = PATH+ os.path.sep+ "excel" + os.path.sep + file

print('file name')

print(FILE_NAME)

    # Reading and filtering Excel file
print('train df')
xdf_data = pd.read_excel(FILE_NAME)

print(xdf_data.head())

# class_names = process_target(target_type=2)
# print(class_names)

xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()

xdf_dset_test = xdf_data[xdf_data["split"] == 'test'].copy()

print(xdf_dset.head())

xdf_dset.loc[:, 'kfold'] = -1

xdf_dset = xdf_dset.sample(frac=1).reset_index(drop=True)

X = xdf_dset.id.values

labels = xdf_dset['target_class'].str.split(',').apply(lambda x: [int(i) for i in x]).tolist()

y = labels

mskf = MultilabelStratifiedKFold(n_splits=5)

for fold, (trn_, val_) in enumerate(mskf.split(X, y)):
    xdf_dset.loc[val_, 'kfold'] = fold




print('after kfold')



print(xdf_dset.tail(5))


def train(fold):
    training_data_path = DATA_DIR

    train_bs = 32
    valid_bs = 16

    df_train = xdf_dset[xdf_dset.kfold != fold].reset_index(drop=True)
    print(df_train.head())

    print('valid df')
    df_valid = xdf_dset[xdf_dset.kfold == fold].reset_index(drop=True)
    print(df_valid.head())


    train_aug = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(p=0.2),
        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value=255.0, always_apply=True),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5),
        A.RandomGamma(p=0.5)

    ])

    valid_aug = A.Compose([
        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value=255.0, always_apply=True)
    ])



    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)


    # Datasets
    partition = {
        'train': list_of_ids,
        'test' : list_of_ids_test
    }

    train_dataset = Dataset(
        list_IDs=partition['train'],
        type_data='train',
        target_type=2,
        xdf_dset=df_train,
        xdf_dset_test=xdf_dset_test,  # Assuming xdf_dset_test is already defined
        aug=train_aug
    )

    valid_dataset = Dataset(
        list_IDs=partition['train'],
        type_data='train',
        target_type=2,
        xdf_dset=df_valid,
        xdf_dset_test=xdf_dset_test,  # Assuming xdf_dset_test is already defined
        aug=valid_aug
    )





    # list_of_ids = list(xdf_dset.id.values)
    # list_of_ids_test = list(xdf_dset_test.id.values)

    # Datasets
    # partition = {
    #     'train': list_of_ids,
    #     'test': list_of_ids_test
    # }


    # train_images = df_train.id.values

    # print('training imgs')
    # train_images = [os.path.join(training_data_path, i) for i in train_images]
    # print(train_images)

    # train_targets = df_train.target_class.apply(lambda x: [int(i) for i in x.split(",")]).tolist()
    # print('train targets')
    # train_targets = df_train.target_class.values
    # print(train_targets)
    # valid_images = df_valid.id.values

    # valid_images = [os.path.join(training_data_path, i) for i in valid_images]

    # valid_targets = df_valid.target_class.apply(lambda x: [int(i) for i in x.split(",")]).tolist()

    # valid_targets = df_valid.target_class.values

    # Define the training dataset
    # train_dataset = Dataset(list_IDs=df_train.id.values, type_data='train', target_type=2, aug=train_aug)
    #
    # # Define the validation dataset
    # valid_dataset = Dataset(list_IDs=valid_images, type_data='train', target_type=2, aug=valid_aug)

    # For training dataset
    # train_dataset = Dataset(list_IDs=train_images, type_data='train', target_type=2, aug=train_aug)

    # train_dataset = Dataset(list_IDs=train_images, type_data='train', target_type=2, aug=train_aug)

    # train_dataset = Dataset(list_IDs=df_train.id.values, type_data='train', target_type=2, xdf_dset=df_train,
    #                         xdf_dset_test=xdf_dset_test, aug=train_aug)

    # For validation dataset
    # valid_dataset = Dataset(list_IDs=valid_images, type_data='train', target_type=2, aug=valid_aug)

    # valid_dataset = Dataset(list_IDs=df_valid.id.values, type_data='train', target_type=2, xdf_dset=df_valid,
    #                         xdf_dset_test=xdf_dset_test, aug=valid_aug)

    # Define the training data loader
    train_data_loader = data.DataLoader(train_dataset, batch_size=train_bs, shuffle=True)

    # Define the validation data loader
    valid_data_loader = data.DataLoader(valid_dataset, batch_size=valid_bs, shuffle=False)


    model = CNN()

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True, factor=0.2)

    criterion = nn.BCEWithLogitsLoss()

    cont = 0
    train_loss_item = []
    valid_loss_item = []
    pred_labels_per_hist = []

    model.phase = 0

    met_valid_best = 0

    for epoch in range(n_epoch):
        train_loss, steps_train = 0, 0

        model.train()
        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
        train_hist = []

        with tqdm(total=len(train_data_loader), desc="Epoch {}".format(epoch)) as pbar:
            for xdata, xtarget in train_data_loader:
                xdata, xtarget = xdata.to(device), xtarget.to(device)

                optimizer.zero_grad()

                output = model(xdata)

                loss = criterion(output, xtarget)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                cont += 1

                steps_train += 1

                train_loss_item.append([epoch, loss.item()])

                pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                if len(pred_labels_per_hist) == 0:
                    pred_labels_per_hist = pred_labels_per
                else:
                    pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                if len(train_hist) == 0:
                    train_hist = xtarget.cpu().numpy()
                else:
                    train_hist = np.vstack([train_hist, xtarget.cpu().numpy()])

                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss / steps_train))

                pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        # Metric Evaluation for Training
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        avg_train_loss = train_loss / steps_train

        # Evaluate on validation data
        valid_loss, steps_valid = 0, 0
        model.eval()
        pred_logits_valid, real_labels_valid = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        with tqdm(total=len(valid_data_loader), desc="Epoch {} - Validation".format(epoch)) as pbar_valid:
            for xdata_valid, xtarget_valid in valid_data_loader:
                xdata_valid, xtarget_valid = xdata_valid.to(device), xtarget_valid.to(device)

                output_valid = model(xdata_valid)

                loss_valid = criterion(output_valid, xtarget_valid)

                valid_loss += loss_valid.item()
                steps_valid += 1

                valid_loss_item.append([epoch, loss_valid.item()])

                pred_logits_valid = np.vstack((pred_logits_valid, output_valid.detach().cpu().numpy()))
                real_labels_valid = np.vstack((real_labels_valid, xtarget_valid.cpu().numpy()))

                pbar_valid.update(1)
                pbar_valid.set_postfix_str("Validation Loss: {:.5f}".format(valid_loss / steps_valid))

        pred_labels_valid = pred_logits_valid[1:]
        pred_labels_valid[pred_labels_valid >= THRESHOLD] = 1
        pred_labels_valid[pred_labels_valid < THRESHOLD] = 0

        # Metric Evaluation for Validation
        valid_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels_valid[1:], pred_labels_valid)

        avg_valid_loss = valid_loss / steps_valid

        # Update scheduler
        scheduler.step(avg_valid_loss)

        # Save model if the current validation metric is better than the best one so far
        if valid_metrics[0] > met_valid_best:
            met_valid_best = valid_metrics[0]

        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.5f} - Validation Loss: {avg_valid_loss:.5f}")



for fold in range(5):
    train(fold)





#
#
#
#
#
#
#
#
#
#
#
#
#









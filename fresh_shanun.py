# This is a sample Python script.
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
torch.cuda.empty_cache()
import os
import albumentations as A
from torchvision.models import resnet
from torchvision.models.resnet import ResNet18_Weights

import gc
gc.collect()
import pretrainedmodels
import torch.nn.functional as F




class ResNet18Custom(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18Custom, self).__init__()
        model = models.resnet18(pretrained=True)

        # Define the layers
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.out = nn.Linear(2048, OUTPUTS_a)

    def forward(self, x, targets=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out
class ResNet18Custom(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18Custom, self).__init__()
        # Load the ResNet18 model with pretrained weights if specified
        model = resnet.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

        # Define the layers
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.out = nn.Linear(512 * 2 * 2, OUTPUTS_a)  # Adjust output size based on the selected ResNet variant

    def forward(self, x, targets=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out

'''
LAST UPDATED 11/10/2021, lsdr
'''

## Process images in parallel

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
print('data path')
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
print(DATA_DIR)
AUGMENTED_DIR = '/home/ubuntu/final_dl/Exam2-v5/augmented/'

sep = os.path.sep


os.chdir(OR_PATH) # Come back to the directory where the code resides , all files will be left on this directory

n_epoch = 1
BATCH_SIZE = 128
LR = 0.001

## Image processing
CHANNELS = 3
IMAGE_SIZE = 400

NICKNAME = "Jeanne"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.2
SAVE_MODEL = True

OUTPUTS_a = 29


def convert_counts_to_format(presence_counts, non_presence_counts):
    result = {}
    for class_name, presence_count in presence_counts.items():
        result[class_name] = {'0': non_presence_counts[class_name], '1': presence_count}
    return result




class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weights, neg_weights):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_weights = pos_weights
        self.neg_weights = neg_weights
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        # Compute BCEWithLogitsLoss for each label
        loss_per_label = self.loss_fn(logits, targets)

        # Apply weights based on the label
        # weighted_loss_per_label = torch.where(targets == 1, loss_per_label * self.pos_weights, loss_per_label * self.neg_weights)
        pos_loss = self.pos_weights * targets * loss_per_label

        # Apply negative weights where target is 0
        neg_loss = self.neg_weights * (1 - targets) * loss_per_label

        total_loss = pos_loss + neg_loss

        # Compute the mean loss across all labels
        loss = torch.mean(total_loss)
        return loss



train_aug = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, always_apply=True),

])









# train_aug = A.Compose([
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, always_apply=True)
# ])



# def count_label_occurrences(df):
#     label_counts = {}
#     for _, row in df.iterrows():
#         target_labels = row['target'].split(',')  # Split target labels
#         target_classes = list(map(int, row['target_class'].split(',')))  # Split target class and convert to int
#         for label, cls in zip(target_labels, target_classes):
#             label_counts[label] = label_counts.get(label, {'0': 0, '1': 0})  # Initialize counts for each label
#             label_counts[label][str(cls)] += 1  # Increment count for 0 or 1
#     return label_counts








#---- Define the model ---- #

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=3)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=3)
        self.convnorm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=3)
        self.convnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=3)
        self.convnorm4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=3)
        self.convnorm5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=3)
        self.convnorm6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, padding=3)
        self.convnorm7 = nn.BatchNorm2d(1024)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, OUTPUTS_a)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.convnorm1(x)
        x = self.pad1(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.convnorm2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.act(x)
        x = self.convnorm3(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.act(x)
        x = self.convnorm4(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.act(x)
        x = self.convnorm5(x)
        x = self.pool(x)

        x = self.conv6(x)
        x = self.act(x)
        x = self.convnorm6(x)
        x = self.pool(x)

        x = self.conv7(x)
        x = self.act(x)
        x = self.convnorm7(x)
        x = self.pool(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x

class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, list_IDs, type_data, target_type, aug = None):
        #Initialization'
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.target_type = target_type
        self.aug = aug

    def __len__(self):
        #Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        #Generates one sample of data'
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
            labels_ohe = [ int(e) for e in y]
        else:
            labels_ohe = np.zeros(OUTPUTS_a)

            for idx, label in enumerate(range(OUTPUTS_a)):
                if label == y:
                    labels_ohe[idx] = 1

        y = torch.FloatTensor(labels_ohe)

        if self.type_data == 'train':
            # print('normal file')
            if "_aug" in xdf_dset.id.get(ID):
                # print('aug file')
                file = AUGMENTED_DIR + xdf_dset.id.get(ID)
                # print(file)
            else:
                file = DATA_DIR + xdf_dset.id.get(ID)
            # print(file)
            #
            # img = cv2.imread(file)
            # img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            # img = self.aug(image=img)['image']




            img = cv2.imread(file)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = self.aug(image=img)['image']

                #
                # img = cv2.imread(file)
                # img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))




        else:
            file = DATA_DIR + xdf_dset_test.id.get(ID)

            img = cv2.imread(file)

            img= cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))

        # if self.aug:
        #     img = self.aug(image=img)['image']

        # Augmentation only for train
        X = torch.FloatTensor(img)

        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))

        return X, y



def read_data(target_type):
    input_paths = []

    # Loop through each ID
    for id in xdf_dset['id']:
        # Check if the ID contains "_aug"
        if "_aug" in id:
            # If it contains "_aug", append the path from AUGMENTED_DIR
            input_paths.append(AUGMENTED_DIR + id)
        else:
            # If not, append the path from DATA_DIR
            input_paths.append(DATA_DIR + id)

    # Convert the list of paths to a numpy array
    ds_inputs = np.array(input_paths)

    ds_targets = xdf_dset['target_class']

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)

    # Datasets
    partition = {
        'train': list_of_ids,
        'test': list_of_ids_test
    }

    # Data Loaders

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True}

    training_set = Dataset(partition['train'], 'train', target_type, aug=train_aug)
    training_generator = data.DataLoader(training_set, **params)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test', target_type)
    test_generator = data.DataLoader(test_set, **params)

    # Make the channel as a list to make it variable

    return training_generator, test_generator



# def read_data(target_type):
#     ## Only the training set
#     ## xdf_dset ( data set )
#     ## read the data data from the file
#
#     ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
#
#
#
#     ds_targets = xdf_dset['target_class']
#
#     # ---------------------- Parameters for the data loader --------------------------------
#
#     list_of_ids = list(xdf_dset.index)
#     list_of_ids_test = list(xdf_dset_test.index)
#
#
#     # Datasets
#     partition = {
#         'train': list_of_ids,
#         'test' : list_of_ids_test
#     }
#
#     # Data Loaders
#
#     params = {'batch_size': BATCH_SIZE,
#               'shuffle': True}
#
#     training_set = Dataset(partition['train'], 'train', target_type, aug=train_aug)
#     training_generator = data.DataLoader(training_set, **params)
#
#     params = {'batch_size': BATCH_SIZE,
#               'shuffle': False}
#
#     test_set = Dataset(partition['test'], 'test', target_type)
#     test_generator = data.DataLoader(test_set, **params)
#
#     ## Make the channel as a list to make it variable
#
#     return training_generator, test_generator



def save_model(model):
    # Open the file

    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))

# def model_definition(pretrained=False):
#     # Define a Keras sequential model
#     # Compile the model
#
#     if pretrained == True:
#         model = models.resnet18(pretrained=True)
#         model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
#     else:
#         model = CNN()
#
#     model = model.to(device)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     criterion = nn.BCEWithLogitsLoss()
#
#     scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)
#
#     save_model(model)
#
#     return model, optimizer, criterion, scheduler
#



def model_definition(pretrained=False):

    if pretrained == True:
        model = ResNet18Custom(pretrained=True)
        # model = models.resnet18(pretrained=True)


        # model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)


    else:
        model = CNN()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Initialize positive and negative weights

    pos_weights = class_weights['positive_weights']
    neg_weights = class_weights['negative_weights']



    pos_weights = torch.tensor([pos_weights[label] for label in pos_weights.keys()])
    neg_weights = torch.tensor([neg_weights[label] for label in neg_weights.keys()])


    # Move weights to device
    pos_weights = pos_weights.to(device)
    neg_weights = neg_weights.to(device)



    # Initialize the custom loss function
    criterion = WeightedBCEWithLogitsLoss(pos_weights, neg_weights)

    criterion_2 = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)

    save_model(model)

    return model, optimizer, criterion, scheduler, criterion_2



def train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on, pretrained=False):
    # Use a breakpoint in the code line below to debug your script.
    model, optimizer, criterion, scheduler, criterion_2 = model_definition(pretrained = False)

    # pos_weights = class_weights['positive_weights']
    # neg_weights = class_weights['negative_weights']
    #
    # # Initialize positive and negative weights
    # pos_weights = torch.tensor([pos_weights[label] for label in pos_weights.keys()])
    # neg_weights = torch.tensor([neg_weights[label] for label in neg_weights.keys()])
    #
    # # Move weights to device
    # pos_weights = pos_weights.to(device)
    # neg_weights = neg_weights.to(device)

    # Initialize the custom loss function
    # weighted_loss_fn = WeightedBCEWithLogitsLoss(pos_weights, neg_weights)
    # weighted_loss_fn_test = WeightedBCEWithLogitsLoss(positive_weights_test, neg)

    cont = 0
    train_loss_item = list([])
    test_loss_item = list([])

    pred_labels_per_hist = list([])

    model.phase = 0

    met_test_best = 0
    for epoch in range(n_epoch):
        train_loss, steps_train = 0, 0

        model.train()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        train_hist = list([])
        test_hist = list([])

        with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:

            for xdata,xtarget in train_ds:

                xdata, xtarget = xdata.to(device), xtarget.to(device)

                optimizer.zero_grad()

                output = model(xdata)

                # Use custom loss function
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
                pbar.set_postfix_str("Test Loss: {:.5f}".format(train_loss / steps_train))

                pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        print('predicted labels')
        pred_labels = pred_logits[1:]
        print(pred_labels)
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        # Metric Evaluation
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        avg_train_loss = train_loss / steps_train

        ## Finish with Training

        ## Testing the model

        model.eval()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        test_loss, steps_test = 0, 0
        met_test = 0

        with torch.no_grad():

            with tqdm(total=len(test_ds), desc="Epoch {}".format(epoch)) as pbar:

                for xdata,xtarget in test_ds:

                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    optimizer.zero_grad()

                    output = model(xdata)

                    # Use custom loss function
                    loss = criterion_2(output, xtarget)

                    test_loss += loss.item()
                    cont += 1

                    steps_test += 1

                    test_loss_item.append([epoch, loss.item()])

                    pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                    if len(pred_labels_per_hist) == 0:
                        pred_labels_per_hist = pred_labels_per
                    else:
                        pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                    if len(test_hist) == 0:
                        tast_hist = xtarget.cpu().numpy()
                    else:
                        test_hist = np.vstack([test_hist, xtarget.cpu().numpy()])

                    pbar.update(1)
                    pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

                    pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                    real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        #acc_test = accuracy_score(real_labels[1:], pred_labels)
        #hml_test = hamming_loss(real_labels[1:], pred_labels)

        avg_test_loss = test_loss / steps_test

        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres +' Train '+met+ ' {:.5f}'.format(dat)


        xstrres = xstrres + " - "
        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test '+met+ ' {:.5f}'.format(dat)
            if met == save_on:
                met_test = dat

        print(xstrres)

        if met_test > met_test_best and SAVE_MODEL:

            torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))
            xdf_dset_results = xdf_dset_test.copy()

            ## The following code creates a string to be saved as 1,2,3,3,
            ## This code will be used to validate the model
            xfinal_pred_labels = []
            for i in range(len(pred_labels)):
                joined_string = ",".join(str(int(e)) for e in pred_labels[i])
                xfinal_pred_labels.append(joined_string)

            xdf_dset_results['results'] = xfinal_pred_labels

            xdf_dset_results.to_excel('results_{}.xlsx'.format(NICKNAME), index = False)
            print("The model has been saved!")
            met_test_best = met_test






#
# def train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on, pretrained = False):
#     # Use a breakpoint in the code line below to debug your script.
#
#     model, optimizer, criterion, scheduler = model_definition(pretrained)
#
#     cont = 0
#     train_loss_item = list([])
#     test_loss_item = list([])
#
#     pred_labels_per_hist = list([])
#
#     model.phase = 0
#
#     met_test_best = 0
#     for epoch in range(n_epoch):
#         train_loss, steps_train = 0, 0
#
#         model.train()
#
#         pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
#
#         train_hist = list([])
#         test_hist = list([])
#
#         with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:
#
#             for xdata,xtarget in train_ds:
#
#                 xdata, xtarget = xdata.to(device), xtarget.to(device)
#
#                 optimizer.zero_grad()
#
#                 output = model(xdata)
#
#                 loss = criterion(output, xtarget)
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item()
#                 cont += 1
#
#                 steps_train += 1
#
#                 train_loss_item.append([epoch, loss.item()])
#
#                 pred_labels_per = output.detach().to(torch.device('cpu')).numpy()
#
#                 if len(pred_labels_per_hist) == 0:
#                     pred_labels_per_hist = pred_labels_per
#                 else:
#                     pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])
#
#                 if len(train_hist) == 0:
#                     train_hist = xtarget.cpu().numpy()
#                 else:
#                     train_hist = np.vstack([train_hist, xtarget.cpu().numpy()])
#
#                 pbar.update(1)
#                 pbar.set_postfix_str("Test Loss: {:.5f}".format(train_loss / steps_train))
#
#                 pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
#                 real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))
#
#         pred_labels = pred_logits[1:]
#         pred_labels[pred_labels >= THRESHOLD] = 1
#         pred_labels[pred_labels < THRESHOLD] = 0
#
#         # Metric Evaluation
#         train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)
#
#         avg_train_loss = train_loss / steps_train
#
#         ## Finish with Training
#
#         ## Testing the model
#
#         model.eval()
#
#         pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
#
#         test_loss, steps_test = 0, 0
#         met_test = 0
#
#         with torch.no_grad():
#
#             with tqdm(total=len(test_ds), desc="Epoch {}".format(epoch)) as pbar:
#
#                 for xdata,xtarget in test_ds:
#
#                     xdata, xtarget = xdata.to(device), xtarget.to(device)
#
#                     optimizer.zero_grad()
#
#                     output = model(xdata)
#
#                     loss = criterion(output, xtarget)
#
#                     test_loss += loss.item()
#                     cont += 1
#
#                     steps_test += 1
#
#                     test_loss_item.append([epoch, loss.item()])
#
#                     pred_labels_per = output.detach().to(torch.device('cpu')).numpy()
#
#                     if len(pred_labels_per_hist) == 0:
#                         pred_labels_per_hist = pred_labels_per
#                     else:
#                         pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])
#
#                     if len(test_hist) == 0:
#                         tast_hist = xtarget.cpu().numpy()
#                     else:
#                         test_hist = np.vstack([test_hist, xtarget.cpu().numpy()])
#
#                     pbar.update(1)
#                     pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))
#
#                     pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
#                     real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))
#
#         pred_labels = pred_logits[1:]
#         pred_labels[pred_labels >= THRESHOLD] = 1
#         pred_labels[pred_labels < THRESHOLD] = 0
#
#         test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)
#
#         #acc_test = accuracy_score(real_labels[1:], pred_labels)
#         #hml_test = hamming_loss(real_labels[1:], pred_labels)
#
#         avg_test_loss = test_loss / steps_test
#
#         xstrres = "Epoch {}: ".format(epoch)
#         for met, dat in train_metrics.items():
#             xstrres = xstrres +' Train '+met+ ' {:.5f}'.format(dat)
#
#
#         xstrres = xstrres + " - "
#         for met, dat in test_metrics.items():
#             xstrres = xstrres + ' Test '+met+ ' {:.5f}'.format(dat)
#             if met == save_on:
#                 met_test = dat
#
#         print(xstrres)
#
#         if met_test > met_test_best and SAVE_MODEL:
#
#             torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))
#             xdf_dset_results = xdf_dset_test.copy()
#
#             ## The following code creates a string to be saved as 1,2,3,3,
#             ## This code will be used to validate the model
#             xfinal_pred_labels = []
#             for i in range(len(pred_labels)):
#                 joined_string = ",".join(str(int(e)) for e in pred_labels[i])
#                 xfinal_pred_labels.append(joined_string)
#
#             xdf_dset_results['results'] = xfinal_pred_labels
#
#             xdf_dset_results.to_excel('results_{}.xlsx'.format(NICKNAME), index = False)
#             print("The model has been saved!")
#             met_test_best = met_test


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
            xmet = -hamming_metric(y_true, y_pred)
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





if __name__ == '__main__':

    # for file in os.listdir(PATH+os.path.sep + "excel"):
    #     if file[-5:] == '.xlsx':
    #         FILE_NAME = PATH+ os.path.sep+ "excel" + os.path.sep + file


    FILE_NAME = '/home/ubuntu/final_dl/Exam2-v5/excel/updated_dataframe_123.xlsx'

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
    xdf_dset = xdf_data[xdf_data["split"] == 'train']

    # print(xdf_dset.tail())

    # xdf_dset = xdf_dset[~xdf_dset['target'].apply(lambda x: 'class1' in x.split(',') and len(x.split(',')) == 1)]

    # print(xdf_dset.tail())

    class_counts_pos = {}
    class_counts_neg = {}

    # Create a set of all possible classes
    all_classes = set()

    for row in xdf_dset.target:
        classes = row.split(',')
        all_classes.update(classes)

    # Initialize counts for all classes to 0
    for class_name in all_classes:
        class_counts_pos[class_name] = 0

    # Update counts for each class
    for row in xdf_dset.target:
        classes = row.split(',')

        for class_name in all_classes:
            if class_name in classes:
                class_counts_pos[class_name] += 1

    for class_name in all_classes:
        class_counts_neg[class_name] = len(xdf_dset) - class_counts_pos[class_name]

    result = convert_counts_to_format(class_counts_pos, class_counts_neg)

    class_weights = {}
    positive_weights = {}
    negative_weights = {}

    for label, counts in result.items():
        num_1s = counts.get('1', 0)
        num_0s = counts.get('0', 0)
        total_samples = num_1s + num_0s

        positive_weights[label] = total_samples / (2 * num_1s) if num_1s != 0 else 0
        negative_weights[label] = total_samples / (2 * num_0s) if num_0s != 0 else 0

    class_weights['positive_weights'] = positive_weights
    class_weights['negative_weights'] = negative_weights

    FILE_NAME_1 = '/home/ubuntu/final_dl/Exam2-v5/excel/train_test.xlsx'

    xdf_data_1 = pd.read_excel(FILE_NAME_1)

    xdf_dset_test= xdf_data_1[xdf_data_1["split"] == 'test'].copy()





    ## read_data creates the dataloaders, take target_type = 2

    train_ds,test_ds = read_data(target_type = 2)

    print('output')
    # OUTPUTS_a = len(class_names)
    #
    # print(OUTPUTS_a)

    list_of_metrics = ['f1_micro']
    list_of_agg = ['avg']

    train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on= 'f1_micro', pretrained=False)
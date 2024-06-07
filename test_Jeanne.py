# This is a sample Python script.
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
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
import os
import argparse

'''
LAST UPDATED 11/10/2021, lsdr
'''

## Process images in parallel

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file


parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
parser.add_argument("--split", default=False, type=str, required=True)  # validate, test, train

args = parser.parse_args()

PATH = args.path
DATA_DIR = args.path + os.path.sep + 'Data' + os.path.sep
SPLIT = args.split


BATCH_SIZE = 128
LR = 0.001
n_epoch = 3

## Image processing
CHANNELS = 3
IMAGE_SIZE = 400

NICKNAME = "Jeanne"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.2

OUTPUTS_a = 29

#---- Define the model ---- #


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


## ------------------ Data Loadaer definition

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
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids_test = list(xdf_dset_test.index)


    # Datasets
    partition = {
        'test' : list_of_ids_test
    }

    # Data Loader

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test', target_type)
    test_generator = data.DataLoader(test_set, **params)

    ## Make the channel as a list to make it variable

    return test_generator


def model_definition():
    # Define a Keras sequential model
    # Compile the model


    model = CNN()

    model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))
    model = model.to(device)

    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))

    criterion = nn.BCEWithLogitsLoss()

    return model, criterion

# def test_model(test_ds, list_of_metrics, list_of_agg , pretrained = False):
#     # Create the test instructions to
#     # Load the model
#     # Create the loop to validate the data
#     # You can use a dataloader
#
#     model, criterion  = model_definition(pretrained)
#     model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))
#
#     pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
#
#     model.eval()
#
#
#     #  Create the evalution
#     #  Run the statistics
#     #  Save the results in the Excel file
#     # Remember to wirte a string con the result (NO A COLUMN FOR each )
#
#
#     xdf_dset_test['results'] = xfinal_pred_labels
#     xdf_dset_test.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)



def test_model(test_ds, list_of_metrics, list_of_agg , pretrained = False):
    # Create the test instructions to
    # Load the model
    # Create the loop to validate the data
    # You can use a dataloader
    cont = 0
    test_loss_item = list([])
    pred_labels_per_hist = list([])

    model, criterion  = model_definition()
    model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))

    for epoch in range(n_epoch):
        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        model.eval()

        test_loss, steps_test = 0, 0
        test_hist = []

        with torch.no_grad():

            with tqdm(total=len(test_ds), desc="Epoch {}".format(epoch)) as pbar:

                for xdata, xtarget in test_ds:

                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    output = model(xdata)

                    loss = criterion(output, xtarget)

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

    avg_test_loss = test_loss / steps_test

    print(test_metrics, avg_test_loss)

    xfinal_pred_labels = []

    for i in range(len(pred_labels)):

        joined_string = ",".join(str(int(e)) for e in pred_labels[i])

        xfinal_pred_labels.append(joined_string)

    xdf_dset_test['results'] = xfinal_pred_labels

    xdf_dset_test.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)




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
            xmet =-hamming_metric(y_true, y_pred)
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

    for file in os.listdir(PATH+os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH+ os.path.sep+ "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    ## Processing Train dataset
    ## Target_type = 1  Multiclass   Target_type = 2 MultiLabel
    # class_names = process_target(target_type = 2)


    FILE_NAME_1 = '/home/ubuntu/final_dl/Exam2-v5/excel/train_test.xlsx'

    xdf_data_1 = pd.read_excel(FILE_NAME_1)


    ## Balancing classes , all groups have the same number of observations
    xdf_dset_test= xdf_data_1[xdf_data_1["split"] == SPLIT].copy()

    ## read_data creates the dataloaders, take target_type = 2

    test_ds = read_data(target_type = 2)

    # OUTPUTS_a = len(class_names)

    list_of_metrics = ['f1_micro']
    list_of_agg = ['avg']

    test_model(test_ds, list_of_metrics, list_of_agg, pretrained=False)
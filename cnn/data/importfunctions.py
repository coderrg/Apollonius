import torch as T 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torchvision.transforms import ToTensor 
import numpy as np 
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import pickle
import time


# MinMax Scaler
def minmaxScale(image):
    minPixel = np.amin(image)
    maxPixel = np.amax(image)
    image = (image - minPixel) / (maxPixel - minPixel)
    return image

def threshold_image(image, threshold=0.1):
    minPixel = np.amin(image)
    maxPixel = np.amax(image)
    gray = (image - minPixel) / (maxPixel - minPixel)
    for i in range(len(gray)):
        for j in range(len(gray[i])):
            gray[i][j] = int((gray[i][j] > threshold)) * 255
    return gray

def generate_windows(image, filename, window_size, interior_size, increment, folder):
    gray_img = threshold_image(image)
    x = 0
    while (x < image.shape[0] - window_size):
        row_has_intersection = False
        y = 0
        while (y < image.shape[1] - window_size):
            window = image[x : x + window_size, y : y + window_size, :]
            has_intersection = False
            
            gray = gray_img[x : x + window_size, y : y + window_size]
            sum_of_grayscale_pixels = np.sum(gray) / 255
            
            # if the image is not completely white, we can add it to the dataset
            if (sum_of_grayscale_pixels < window_size * window_size) and sum_of_grayscale_pixels >= window_size * window_size/10:
                window_id = str(x) + "_" + str(y) + "_" + filename
                cv2.imwrite(os.path.join(folder, window_id), gray)
                
            if (has_intersection):
                y += int(increment / 5)
            else:
                y += increment
        if (row_has_intersection):
            x += int(increment / 5)
        else:
            x += increment


class CNN(nn.Module):
    def __init__(self, lr, epochs, input_dims, batch_size, name, best_name, chkpt_dir, dataset_prefix, dataset_suffix,
                     num_classes=2):
        super(CNN, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name)
        self.best_checkpoint_file = os.path.join(self.chkpt_dir, best_name)
        self.epochs = epochs
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size 
        self.num_classes = num_classes
        self.loss_history = []
        self.acc_history = []
        self.skipped_points = []
        self.val_history = [0, 0, 0]
        self.device = T.device("cpu")
        
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.bn6 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)

        fc_dims = self.calc_fc_dims()
        self.fc1 = nn.Linear(fc_dims, 32)
        self.fc2 = nn.Linear(32, 16) 
        self.fc3 = nn.Linear(16, self.num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr) # the self.parameters() comes from nn.Module 

        self.loss = nn.CrossEntropyLoss() # since we have more than 2 classes cross entropy is best otherwise with 2 classes we might be able to use binary loss
        
        self.to(self.device)
        self.get_data(dataset_prefix + "train" + dataset_suffix, dataset_prefix + "val" + dataset_suffix, dataset_prefix + "test" + dataset_suffix)

    def calc_fc_dims(self):
        batch_data = T.zeros((1, 1, 25, 25)) # 4-tensor of 0s and we plug it into the layer and see what comes out 
        batch_data = self.conv1(batch_data)
        # batch_data = self.bn1(batch_data) # batch norm layer does not change dimesnionality
        batch_data = self.conv2(batch_data)
        # batch_data = self.bn2(batch_data)
        batch_data = self.conv3(batch_data)
        # batch_data = self.bn3(batch_data)
        batch_data = self.maxpool1(batch_data)
        batch_data = self.conv4(batch_data)
        batch_data = self.conv5(batch_data)
        batch_data = self.conv6(batch_data)
        batch_data = self.maxpool2(batch_data)

        return int(np.prod(batch_data.size())) # this will give us the input dimesniosn 

    def forward(self, batch_data): # NOTE: we can kinda combine this with calc_fc_dims() 
        batch_data = T.tensor(batch_data).to(self.device) # lower case tensor() preserves the datatype while Tensor() changes the datatype to some default datatype 
        # we do a to(self.device) in order to make sure it is not a cuda tensor 
        try: 
            batch_data = T.reshape(batch_data, self.input_dims)
        except:
            pass
        batch_data = batch_data.type(T.FloatTensor)
#             print(batch_data.size(), batch_data.type(), batch_data)
# 		print(batch_data.type())
        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data) # debate about whether to do batch norm before or after relu but this works fine for this one especially since relu is a noncommutative operation with respect to things like addition
        batch_data = F.relu(batch_data)

        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool1(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = F.relu(batch_data) 

        batch_data = self.conv5(batch_data)
        batch_data = self.bn5(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv6(batch_data)
        batch_data = self.bn6(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool2(batch_data)

        batch_data = batch_data.view(batch_data.size()[0], -1)

        batch_data = self.fc1(batch_data)
        batch_data = self.fc2(batch_data)
        classes = self.fc3(batch_data)
        # note that we are not doing another activation after this since the linear cross entropy loss performs a softmax activation on it already 

        return classes
        
    def get_data(self, train_filename, val_filename, test_filename):

        with open(train_filename, 'rb') as f:
            train_dataset = pickle.load(f)
        with open(val_filename, 'rb') as f:
            val_dataset = pickle.load(f)
        with open(test_filename, 'rb') as f:
            test_dataset = pickle.load(f)
        
#         mnist_train_data = MNIST("mnist", train=True, download=True, 
#                                 transform=ToTensor())
        self.train_data_loader = T.utils.data.DataLoader(train_dataset, 
                                    batch_size=self.batch_size, shuffle=True, # always want to shuffle the in case it was not preshuffled so that we get actual learning 
                                    num_workers=3) # this part is just so that the computer can split up the task so make it less than 4 for a mac 
        self.val_data_loader = T.utils.data.DataLoader(val_dataset, 
                                    batch_size=1, shuffle=True, 
                                    num_workers=3)
#         mnist_test_data = MNIST("mnist", train=False, download=True, 
#                                 transform=ToTensor())
        self.test_data_loader = T.utils.data.DataLoader(test_dataset, 
                                    batch_size=self.batch_size, shuffle=True, # always want to shuffle the in case it was not preshuffled so that we get actual learning 
                                    num_workers=3) 

    def _val(self):
        y_true = []
        y_pred = []
        for j, (input, label) in enumerate(self.val_data_loader): 
            input = T.reshape(input, (1,1,25,25))
#             print(input.type())
            prediction = self.forward(input)
            prediction = F.softmax(prediction, dim=1)
            classes = T.argmax(prediction, dim=1)
            y_pred.append(classes.item())
            y_true.append(label.item())
#         print("y_true is: \n", y_true[:10], "\n ---------------------------- \n ", "y_pred is: \n", y_pred[:10])
        conf_matrix = confusion_matrix(y_true, y_pred, labels=[0,1])
        print("The confusion matrix is: \n", conf_matrix)
        self.val_history.append(f1_score(y_true, y_pred, zero_division=1))
        return conf_matrix[0][0] + conf_matrix[1][1]/(np.sum(conf_matrix))
        
        
    def _train(self):
        self.train() # this is important if you are using pytorch with batch norm (it only switches the neural net to a train mode where it remembers the batch norm statistics for training thus only do this with batch norm)
        for i in range(self.epochs): # iteration over the full dataset (we have 60,000 in training set, 10,000 in the test set) so we want to iterate over it many many times 
            ep_loss = 0
            ep_acc = [] # this is epoch accuracy 
            counter = 0
            for j, (input, label) in enumerate(self.train_data_loader): # the default format is an integer and a tuple with an input and an actual label
#                 print(input.type())
                self.optimizer.zero_grad() # remember to always zero the gradient before your training as otherwise it will remember stuff from the last cycle 
                label = label.to(self.device)
#                 print(input.size(), label)
#                 print(input.type(), label.type())
                if input.size()[0] != 8:
                    print("I am passing")
                    counter += 1
                else:
                    prediction = self.forward(input)
                    loss = self.loss(prediction, label)
                    prediction = F.softmax(prediction, dim=1) # the softmax is so that we get a probabilities over the classes 
                    classes = T.argmax(prediction, dim=1)
                    wrong = T.where(classes != label, T.tensor([1.]).to(self.device), T.tensor([0.]).to(self.device)) # this looks at when the labels are not correct and marsk those with a 1
                    acc = 1 - T.sum(wrong) / self.batch_size
                    
                    ep_acc.append(acc.item()) # acc is a tensor so we look at the item in the tensor 
                    self.acc_history.append(acc.item())
                    ep_loss += loss.item()
                    loss.backward() # this calculates the gradient and is VERY IMPORTANT 
                    self.optimizer.step() # this uses the optimizer to adjust the weights ALSO VERY IMPORTANT 

                    if (j % 1000 == 0):
                        print("Epoch ", i, "Data Point ", j, "total loss %.3f" % ep_loss, "accuracy %.3f" % np.mean(ep_acc))

            self._val()
            print("Finished Epoch ", i, "total loss %.3f" % ep_loss, "accuracy %.3f" % np.mean(ep_acc), 
                  "validation f1 %.3f" % self.val_history[-1])
            print(self.val_history)
            self.skipped_points.append(counter)
            if counter:
                print("Number of skipped points is ", counter)
            self.loss_history.append(ep_loss)
#             if (i % 1 == 0):
            if self.val_history[-1] >= self.val_history[-2]:
                self.save_checkpoint(best=self.val_history[-1] == max(self.val_history))
#             if self.val_history[-1] < self.val_history[-2] and self.val_history[-2] < self.val_history[-3]:
#                 break
    def _test(self):
        
        # self.test() # this is important if you are using pytorch with batch norm so now we can run it in test mode
        ep_loss = 0
        ep_acc = [] # this is epoch accuracy 
        counter = 0
        y_true = []
        y_pred = []
        for j, (input, label) in enumerate(self.test_data_loader): # the default format is an integer and a tuple with an input and an actual label
            label = label.to(self.device)
            if input.size()[0] != 8:
#                 print("I am passing")
                counter += 1
            else:  
                prediction = self.forward(input)
                loss = self.loss(prediction, label)
                prediction = F.softmax(prediction, dim=1) # the softmax is so that we get a probabilities over the classes 
                classes = T.argmax(prediction, dim=1)
                wrong = T.where(classes != label, T.tensor([1.]).to(self.device), T.tensor([0.]).to(self.device)) # this looks at when the labels are not correct and marsk those with a 1
                acc = 1 - T.sum(wrong) / self.batch_size
                for i in range(self.batch_size):
                    y_true.append(label[i].item())
                    y_pred.append(classes[i].item())
                ep_acc.append(acc.item()) # acc is a tensor so we look at the item in the tensor 
                ep_loss += loss.item()
        print("The confusion matrix is: \n", confusion_matrix(y_true, y_pred, labels=[0,1]))
        print(f"The f1 score is {f1_score(y_true, y_pred, zero_division=1)}")

        print("Total loss %.3f" % ep_loss, "accuracy %.3f" % np.mean(ep_acc))
        if counter:
            print("Number of skipped points is ", counter)
        
        return ep_acc

    def save_checkpoint(self, filename=None, best=False):
        filename = self.checkpoint_file if filename is None else filename
        print('... saving checkpoint ...')
        T.save(self.state_dict(), filename)
        if best:
            T.save(self.state_dict(), self.best_checkpoint_file)

    def load_checkpoint(self, filename=None):
        filename = self.best_checkpoint_file if filename is None else filename
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(filename))
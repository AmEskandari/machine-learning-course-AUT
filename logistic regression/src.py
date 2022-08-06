import torch

import pandas as pd
import numpy as np


from typing import Counter
from tqdm import tqdm
from torch import norm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader
from scipy.stats import multivariate_normal



class Data:
    """ Data class for reading, pre-processing and converting the type of
        raw data  """

    def __init__(self,data_dir, random_seed):
        """ Creat new Data instance
            
            radnom_seed, seed for random generation in order to have reproducible results
            data_dir direction of raw data, string type
    
        """
        rng = np.random.RandomState(random_seed)

        raw_data =  shuffle(pd.read_csv(data_dir).drop(['Genus','Species','RecordID'], axis = 1), 
                            random_state = rng) 
        self._x, self._y = raw_data.iloc[:,:-1].to_numpy(),raw_data.iloc[:,-1] #data
        self._class_dict = defaultdict(int)     

        self._fold_index = 0

    def encode_labels(self):
        """One-hot encoding of categorical label"""

        label_enc = LabelEncoder()
        y_train_numeric = label_enc.fit_transform(self._y)
        self._y = np.zeros((y_train_numeric.size, y_train_numeric.max()+1))
        self._y[np.arange(y_train_numeric.size),y_train_numeric]  = 1

        for idx, classes in enumerate(label_enc.__dict__['classes_']):
            self._class_dict[classes] = idx

    def splite_data(self,train_ratio,validation_ratio, per_class = None):
        """ This function outputs train,test,validation data
            in a Dataset class type which will be used for 
            training and evaluation the model. Order
            train data, validation data, test data"""
        if ~bool(per_class):
            
            train_obj = Dataset(torch.FloatTensor(self._x[:int(self._x.shape[0]*train_ratio)]),
                            torch.FloatTensor(self._y[:int(self._y.shape[0]*train_ratio)]))
            valid_obj = Dataset(torch.FloatTensor(self._x[int(self._x.shape[0]*train_ratio):int(self._x.shape[0]*(train_ratio+validation_ratio))]),
                            torch.FloatTensor(self._y[int(self._y.shape[0]*train_ratio):int(self._y.shape[0]*(train_ratio+validation_ratio))]))
            test_obj = Dataset(torch.FloatTensor(self._x[int(self._x.shape[0]*(train_ratio+validation_ratio)):]),
                           torch.FloatTensor(self._y[int(self._y.shape[0]*(train_ratio+validation_ratio)):]))
        else:

            x_train, x_test , y_train, y_test = train_test_split(self._x,self._y, test_size = (1-train_ratio), stratify=self._y)

            train_obj = Dataset(torch.FlaotTensor(x_train,y_train))

            test_obj = Dataset(torch.FlaotTensor(x_test,y_test))

            valid_obj = None                           

        return train_obj, valid_obj, test_obj
    
    def _splite_data_cross_validation(self,k):
        fold_obj = KFold(n_splits=k,shuffle=True)
        self._k_flag = 0
        self._fold_index = list(fold_obj.split(self._x))
    
    def reset_fold(self,k):
        self._splite_data_cross_validation(k)

    def get_fold(self,k):
        
        if isinstance(self._fold_index, int):
            self._splite_data_cross_validation(k)
        
        if self._k_flag == k:
            raise ValueError('There is no fold anymore')        
        train_idx, test_idx = self._fold_index[self._k_flag][0],self._fold_index[self._k_flag][1], 

        x_train, x_test = self._x[train_idx], self._x[test_idx]
        y_train, y_test = self._y[train_idx], self._y[test_idx]

        train_obj = Dataset(torch.FloatTensor(x_train),torch.FloatTensor(y_train))
        test_obj = Dataset(torch.FloatTensor(x_test),torch.FloatTensor(y_test))
        self._k_flag += 1

        return train_obj, test_obj


    def get_label_code(self):
        """ This method outputs a dictionary that indicate the 
            order of labels in one-hot coding"""
        return self._class_dict
    
    def get_num_classes(self):
        """ This method returns the number of label classes."""
        return self._y.shape[1]

    def get_num_dim_features(self):
        """ This method returns the dimention of inpute data."""
        return self._x.shape[1]


class Dataset:
    """ Dataset class for PyTorch DataLoader."""    

    def __init__(self,x,y):
        """ Creat new Dataset instance

            x data features, float32 tensor
            y data labels, float32 tensor
        
        """
        if (x.dtype == torch.float32) and (y.dtype == torch.float32):
            self._x = x
            self._y = y 
        else:
            raise TypeError('x or y or both must be torch.float32 data type') 
        
    def __len__(self):
        return self._x.shape[0]
            
    def __getitem__(self,idx):
        return self._x[idx], self._y[idx]

class LogesticRegressionClf:
    """ Logistic Regression class for training and evaluating the Logistic Rgression
        classifier, this class is capabalte of handeling multi-class classification
        problems."""

    def __init__(self, number_of_classes, feature_dim):
        """ Craet new instance of Logistic Regression Class
        
            number_of_classes, integer"""

        self._num_class = number_of_classes # to be or not to be
        self._feature_dim = feature_dim # to be or not to be
        self._w = torch.normal(mean = 0, std = 1/(feature_dim) ,
                               size = [1+feature_dim,number_of_classes], requires_grad = True) 


        self._var_nb = np.zeros(feature_dim, dtype=np.float32)
        self._mean_preclass_nb = np.zeros((number_of_classes, feature_dim),  dtype=np.float32)
        self._class_dist_nb = 0


    def fit(self, dataset, epoch_num, learning_rate, device, gama = None, batch_size = None):
        """ This method trains the algorithm
        
            dataset, the dataset obejct ready to load by DataLoader
            epoch_num, number of epochs which the algorithm iterate on the dataset
            learning_rate, hyperparameter to control the parametrs change pace
            device, 'gpu' or 'cpu'
            gama, the hyperparameter to regularize the algorithm's cost function
            batch_size, number of samples in each gradient update. 
            
            """

        b_size = batch_size if bool(batch_size) else len(dataset) 
        data_loader = DataLoader(dataset, batch_size = b_size)


        for epoch in tqdm(range(epoch_num)):
            accuracy = 0
            loss_epoch = []            
            for x,y in data_loader:
                unit_tensor = torch.FloatTensor([1.0]*x.shape[0]).reshape(x.shape[0],-1)
                logit_matrix = torch.cat([x,unit_tensor], axis=1).to(device) @ self._w.to(device)
                y_hat = torch.zeros((int(x.shape[0]),self._num_class)).to(device)
  
                for row in range(int(x.shape[0])):
                    denominator = 1. + torch.sum(torch.exp(logit_matrix[row])) 
                    for col in range(self._num_class):
                        y_hat[row,col] = torch.exp(logit_matrix[row,col])/denominator if col != (self._num_class-1) else 1/denominator
   
                    if torch.argmax(y_hat[row]) == torch.argmax(y[row]):
                       accuracy += 1     
                if bool(gama):
                    loss = self._loss(y_hat,y) + gama * norm(self._w, p = 2) 
                else:
                    loss = self._loss(y_hat,y) 

                loss_epoch.append(loss.item())
                loss.backward()

                with torch.no_grad():
                    self._w = self._w - learning_rate * self._w.grad
                    self._w.requires_grad = True
            
            #print(f"Epoch Number : {epoch + 1} | Epoch Loss : {sum(loss_epoch):.2f} | Epoch Accuracy : {100*(accuracy / len(dataset)):.2f}")
        
    
    def evalute(self, dataset, device):
        """ This method evalutes the tarined algorithm
        
            dataset, the dataset obejct ready to load by DataLoader
            device, 'gpu' or 'cpu'
        
            return, the accuracy of the inpute dataset
        """
        
        b_size =  len(dataset) 
        data_loader = DataLoader(dataset, batch_size = b_size)


        accuracy = 0            
        
        x,y = next(iter(data_loader))
        
        unit_tensor = torch.FloatTensor([1.0]*x.shape[0]).reshape(x.shape[0],-1)
        logit_matrix = torch.cat([x,unit_tensor], axis=1).to(device) @ self._w.to(device)
        
        y_hat = torch.zeros((int(x.shape[0]),self._num_class)).to(device)
  
        for row in range(int(x.shape[0])):
            denominator = 1. + torch.sum(torch.exp(logit_matrix[row])) 
            for col in range(self._num_class):
                y_hat[row,col] = torch.exp(logit_matrix[row,col])/denominator if col != (self._num_class-1) else 1/denominator
            
            if torch.argmax(y_hat[row]) == torch.argmax(y[row]):
                accuracy += 1     

        return (accuracy / len(dataset)) * 100
    
    def predict(self, x_inpute, device):
        """ This method return the predicted label for inpute data instances

            x_inpute, inpute instance. Could be one or multiple isntance
            device, 'cpu', 'gpu'

            return, the predicted label in a vector
        """
        x_inpute = x_inpute.reshape(-1, self._feature_dim)

        logit_matrix = x_inpute.to(device) @ self._w.to(device)
        y_hat = torch.zeros((int(x_inpute.shape[0]),self._num_class)).to(device)        
        y_predicted = torch.zeros(int(x_inpute.shape[0])).to(device)       

        for row in range(int(x_inpute.shape[0])):
            denominator = 1. + torch.sum(torch.exp(logit_matrix[row])) 
            for col in range(self._num_class):
                y_hat[row,col] = torch.exp(logit_matrix[row,col])/denominator if col != (self._num_class-1) else 1/denominator
   
            y_predicted[row] = torch.argmax(y_hat[row])
        
        return y_predicted
    
    def fit_nb(self, dataset):
        """ This method fits the parameters of the algorithm by
            means of naive bayse assumption"""
            
        b_size = len(dataset)
        data_loader = DataLoader(dataset, batch_size = b_size)            
        
        x,y = next(iter(data_loader))
        
        x = x.numpy()
        y = y.numpy()

        label_cnt = Counter()
        label_cnt.update(np.argmax(y, axis=1))
        self._class_dist_nb = {k:v/sum(label_cnt.values()) for k,v in label_cnt.items()}

        for col_idx in range(self._feature_dim):
            self._var_nb[col_idx] = np.var(x[:,col_idx])

        for label_idx in range(self._num_class):
            for col_idx in range(self._feature_dim):
                self._mean_preclass_nb[label_idx, col_idx] = np.mean(x[np.argmax(y, axis=1) == label_idx,col_idx])
    
        self._w = self._w.detach().numpy()

        for label_idx in range(self._num_class):
            for feature_idx in range(self._feature_dim+1):
                
                if feature_idx == self._feature_dim:
                    term_2 = (self._mean_preclass_nb[0]**2 - self._mean_preclass_nb[label_idx]**2)/self._var_nb**2 
                    self._w[feature_idx, label_idx] = np.log(self._class_dist_nb[label_idx]/self._class_dist_nb[label_idx]) + np.sum(term_2)
                else:
                    self._w[feature_idx, label_idx] = (self._mean_preclass_nb[label_idx,feature_idx] - self._mean_preclass_nb[0,feature_idx])/self._var_nb[col_idx] ** 2 
        

        self._w = torch.FloatTensor(self._w)


    def _loss(self,y_hat, y_true):
        """ This method calculates the negetive log likelihood
            
            y_hat , Calculated matrix by each class parameters. The row is for a one sample.
            y_true, One-hot matrix label. In each row the associated columns to the label are 1.

            """
        
        return -(torch.sum(torch.log(torch.diag(y_hat @ y_true.T)))) 
    
    def reset_parameters(self, generate):        
        """ This method resets the parameters values """
    
        self._w = torch.normal(mean = 0, std = 1/(self._feature_dim) ,
                               size = [1+self._feature_dim, self._num_class], requires_grad = True,
                               generator = generate) 


class GuassianNaiveBayse:
    """ Guassian Naive Bayes classifier algorithm """
    
    def __init__(self, train_dataset):
        """ Creat new instance of GNB algorith
            
            train_dataset, inpute dataset object"""   
        
        b_size =  len(train_dataset) 
        data_loader = DataLoader(train_dataset, batch_size = b_size)            
        
        x,y = next(iter(data_loader))
        
        label = torch.eye(y.shape[1])
       
        self._label_var = {}
        self._label_mean = {}
        
        for idx, label_vec in enumerate(iter(label[:])):
            self._label_var[idx] = torch.cov(x[torch.all(y[:] == label_vec, axis = 1),:].T).diagonal() 
            self._label_mean[idx] = torch.mean(x[torch.all(y[:] == label_vec, axis = 1),:], axis = 0)
        

        self._model_obj = defaultdict()
        
        for idx in range(int(label.shape[0])):
            self._model_obj[idx] = multivariate_normal((self._label_mean[idx]).numpy(),
                                np.diag((self._label_var[idx]).numpy()), allow_singular = True)
                                                        
        self._prior_term = multivariate_normal(5,1).rvs(size=1)
    
    def _predict(self, instance):

        prd_y = np.zeros(len(self._model_obj))
        for idx, dist in self._model_obj.items():
        
            prd_y[idx] = np.log(dist.pdf(instance))+ np.log(self._prior_term) 
        return np.argmax(prd_y)
        
    def evalute(self, data):
        """ This method return the accuracy of algorithm""" 
        
        b_size =  len(data) 
        data_loader = DataLoader(data, batch_size = b_size)            
        
        x,y = next(iter(data_loader))
        x = x.numpy()

        y_predict = torch.zeros(b_size) 
        
        accuracy = 0
        for idx in range(b_size):
            y_predict[idx] = self._predict(x[idx])
            
            if torch.argmax(y[idx]) == y_predict[idx]:
                accuracy += 1 
                    
        accuracy = (accuracy / b_size) * 100

        return y_predict, accuracy


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 
import math

from main import confusion_matrix, train_test_split, part_d
from tqdm import tqdm
from seaborn import heatmap
from scipy.stats import multivariate_normal



classes = ['Leptodactylidae' ,'Hylidae', 'Dendrobatidae' ,'Bufonidae']
classes_dic = {0:'Leptodactylidae',1:'Hylidae',2:'Dendrobatidae',3:'Bufonidae'}
classes_dic_inv = {'Leptodactylidae':0,'Hylidae':1,'Dendrobatidae':2,'Bufonidae':3}


def confusion_matrix(y_true,y_prd):
    matrix = np.zeros((np.unique(y_true).size,np.unique(y_true).size),dtype=int)
    y_true = np.array(y_true)
    y_prd = np.array(y_prd)
    
    for idx in range(y_true.size):

        matrix[int(classes_dic_inv[y_true[idx]]),int(classes_dic_inv[y_prd[idx]])] += 1
    
  
    return matrix


def confidence_matrix(predition_out,prediction_labels,true_labels):
    """
    pediction_out -> a list thats contains probabiliy of belonging to each calss
    prediction_labels -> the predicted label
    true_labels -> the ground truth label probability

    """
    
    matrix = np.zeros((np.unique(true_labels).size,np.unique(true_labels).size),dtype=float)
    conf_matrix = confusion_matrix(true_labels,prediction_labels)

    for idx in range(true_labels.shape[0]):
        predition_out[idx].sort()
        matrix[int(classes_dic_inv[true_labels.iloc[idx]]),int(classes_dic_inv[prediction_labels[idx]])] += (predition_out[idx][0] - predition_out[idx][1])

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if conf_matrix[i,j] != 0:
                matrix[i,j] = matrix[i,j] / conf_matrix[i,j]

    return matrix

def bounous_part_3(part,data):
    
    

    fold_lentgh = int(data.shape[0]/4)

    accuracy_training_all = []
    accuracy_testing_all = []

    for fold in range(4):


        all_indexs = list(range(data.shape[0]))


        test_idx = list(range(fold_lentgh*fold,fold_lentgh*(fold+1)))
        train_idx = list(set(all_indexs)-set(test_idx))
        
        train,test = data.iloc[train_idx,:],data.iloc[test_idx,:]

        x_train, y_train = train.iloc[:,:-1], train.iloc[:,-1]
        x_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]
        

        if part == 'v2':
            p_training = [p/(sum(list(train['Family'].value_counts()))) for p in list(train['Family'].value_counts())] 
            p_testing = [p/(sum(list(test['Family'].value_counts()))) for p in list(test['Family'].value_counts())] 

        else:
            p_training = [0.25,0.25,0.25,0.25]
            p_testing = [0.25,0.25,0.25,0.25]

        means = []
        covs = []
    
        for label in classes:
            
            means.append(np.array([np.mean(x_train.loc[train['Family'] == label,col]) for col in list(x_train.columns)]))

            covs.append((np.cov(x_train.loc[train['Family'] == label].T)*np.eye(means[0].shape[0])))
            

        normal_dist_Leptodactylidae = multivariate_normal(means[0],covs[0], allow_singular = True)
        normal_dist_Hylidae = multivariate_normal(means[1],covs[1], allow_singular = True)
        normal_dist_Dendrobatidae = multivariate_normal(means[2],covs[2], allow_singular = True)
        normal_dist_Bufonidae = multivariate_normal(means[3],covs[3], allow_singular = True)


        pred_traning = []
        correct_training = 0
        for idx in tqdm(range(x_train.shape[0]), desc = 'Training Phase'):
        
            out = [normal_dist_Leptodactylidae.logpdf(x_train.iloc[idx])+np.log(p_training[0]),
                   normal_dist_Hylidae.logpdf(x_train.iloc[idx])+np.log(p_training[1]),
                   normal_dist_Dendrobatidae.logpdf(x_train.iloc[idx])+np.log(p_training[2]) ,
                   normal_dist_Bufonidae.logpdf(x_train.iloc[idx])+np.log(p_training[3]) ]
            
            pred_traning.append(classes_dic[out.index(max(out))])
        
            if y_train.iloc[idx] == classes_dic[out.index(max(out))]:
                correct_training += 1

    
        pred_testing = []
        correct_testing  = 0
        for idx in tqdm(range(x_test.shape[0]), desc = 'Testing Phase'):
        
            out = [normal_dist_Leptodactylidae.logpdf(x_test.iloc[idx])+np.log(p_testing[0]),
                   normal_dist_Hylidae.logpdf(x_test.iloc[idx])+np.log(p_testing[1]),
                   normal_dist_Dendrobatidae.logpdf(x_test.iloc[idx])+np.log(p_testing[2]),
                   normal_dist_Bufonidae.logpdf(x_test.iloc[idx])+np.log(p_testing[3])]
            
            pred_testing.append(classes_dic[out.index(max(out))])
             
            if y_test.iloc[idx] == classes_dic[out.index(max(out))]:
                correct_testing += 1

    
        accuracy_training_all.append(100*(correct_training/x_train.shape[0]))
        accuracy_testing_all.append(100*(correct_testing/x_test.shape[0]))
    
        print('')
        print(f'Training Accuracy : {100*(correct_training/x_train.shape[0]):.2f} | Testing Accuracy : {100*(correct_testing/x_test.shape[0]):.2f} \n')

        confusion_matrix_training = confusion_matrix(y_train,pred_traning)
        confusion_matrix_testing = confusion_matrix(y_test,pred_testing)

        df_training = pd.DataFrame(confusion_matrix_training, index = classes,
                  columns = classes)
    
        df_testing = pd.DataFrame(confusion_matrix_testing, index = classes,
                  columns = classes)


        plt.figure(figsize = (14,5))
    
        plt.subplot(1,2,1)

        plt.title(f'Training data confusion matrix fold {fold+1}')

        heatmap(df_training, annot = True, fmt="d")

        plt.subplot(1,2,2)

        plt.title(f'Testing data confusion matrix fold {fold+1}')

        heatmap(df_testing, annot = True, fmt="d")

        plt.savefig(os.getcwd()+ f'/part_d_conf_matrix_fold_{fold+1}_{part}.jpg', dpi = 400)

        print(' ')
        print(' ')
    
        #plt.show()

        print('')
    print(f'Mean of Training Accuracy : {(sum(accuracy_training_all)/len(accuracy_training_all)):.2f} | Mean of Testing Accuracy : {(sum(accuracy_testing_all)/len(accuracy_testing_all)):.2f}')
    print('')


def bounous_part_2(part):


    accuracy_training_all = []
    accuracy_testing_all = []

    for iteration,seed in enumerate([42,52]):

        train,test = train_test_split(data, test_size=0.3)

        x_train, y_train = train.iloc[:,:-1], train.iloc[:,-1]
        x_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]

        if part == 'b':
            p_training = [p/(sum(list(train['Family'].value_counts()))) for p in list(train['Family'].value_counts())] 
            p_testing = [p/(sum(list(test['Family'].value_counts()))) for p in list(test['Family'].value_counts())] 

        else:
            p_training = [0.25,0.25,0.25,0.25]
            p_testing = [0.25,0.25,0.25,0.25]

        means = []
        covs = []
    
        for label in classes:
            
            means.append(np.array([np.mean(x_train.loc[train['Family'] == label,col]) for col in list(x_train.columns)]))
           
            w = np.cov(x_train.loc[train['Family'] == label].T)
            covs.append(w)
        
               

        normal_dist_Leptodactylidae = multivariate_normal(means[0],covs[0], allow_singular = True)
        normal_dist_Hylidae = multivariate_normal(means[1],covs[1],allow_singular = True)
        normal_dist_Dendrobatidae = multivariate_normal(means[2],covs[2],allow_singular = True)
        normal_dist_Bufonidae = multivariate_normal(means[3],covs[3],allow_singular = True)


        pred_traning = []
        pred_traning_probability = []
        correct_training = 0
        for idx in tqdm(range(x_train.shape[0]), desc = 'Training Phase'):
        
            out = [normal_dist_Leptodactylidae.pdf(x_train.iloc[idx]),
                   normal_dist_Hylidae.pdf(x_train.iloc[idx]),
                   normal_dist_Dendrobatidae.pdf(x_train.iloc[idx]) ,
                   normal_dist_Bufonidae.pdf(x_train.iloc[idx])]
          
  
            pred_traning.append(classes_dic[out.index(max(out))])
        
            if y_train.iloc[idx] == classes_dic[out.index(max(out))]:
                correct_training += 1

    
        pred_testing = []
        pred_testing_probability = []
        correct_testing  = 0
        for idx in tqdm(range(x_test.shape[0]), desc = 'Testing Phase'):
        
            out = [normal_dist_Leptodactylidae.pdf(x_test.iloc[idx])+np.log(p_testing[0]),
                   normal_dist_Hylidae.pdf(x_test.iloc[idx])+np.log(p_testing[1]),
                   normal_dist_Dendrobatidae.pdf(x_test.iloc[idx])+np.log(p_testing[2]),
                   normal_dist_Bufonidae.pdf(x_test.iloc[idx])+np.log(p_testing[3])]
            
            pred_testing_probability.append([np.exp(item) for item in out])

            pred_testing.append(classes_dic[out.index(max(out))])
             
            if y_test.iloc[idx] == classes_dic[out.index(max(out))]:
                correct_testing += 1

    
        accuracy_training_all.append(100*(correct_training/x_train.shape[0]))
        accuracy_testing_all.append(100*(correct_testing/x_test.shape[0]))
    
        print('')
        print(f'Training Accuracy : {100*(correct_training/x_train.shape[0]):.2f} | Testing Accuracy : {100*(correct_testing/x_test.shape[0]):.2f} \n')

        confidence_matrix_training = confidence_matrix(pred_traning_probability,pred_traning,y_train)
        confidence_matrix_testing = confidence_matrix(pred_testing_probability,pred_testing,y_test)

        df_training = pd.DataFrame(confidence_matrix_training, index = classes,
                  columns = classes)
    
        df_testing = pd.DataFrame(confidence_matrix_testing, index = classes,
                  columns = classes)


        plt.figure(figsize = (14,5))
    
        plt.subplot(1,2,1)

        plt.title(f'Training data confusion matrix iteration {iteration+1}')

        heatmap(df_training, annot = True, fmt="d")

        plt.subplot(1,2,2)

        plt.title(f'Testing data confusion matrix iteration {iteration+1}')

        heatmap(df_testing, annot = True, fmt="d")

#        plt.savefig(os.getcwd()+ f'/conf_matrix_iteraion_{iteration+1}_{part}.jpg', dpi = 400)

        print(' ')
        
    
        plt.show()

    print('')
    print(f'Mean of Training Accuracy : {(sum(accuracy_training_all)/len(accuracy_training_all)):.2f} | Mean of Testing Accuracy : {(sum(accuracy_testing_all)/len(accuracy_testing_all)):.2f}')
    print('')


if __name__ == '__main__':
    data = pd.read_csv(os.getcwd()+'/Frogs_MFCCs.csv').drop(['MFCCs_18','MFCCs_19','MFCCs_20','MFCCs_21','MFCCs_22','Genus', 'Species', 'RecordID'], axis=1).sample(frac=1)
    bounous_part_3('v1',data)

    bounous_part_3('v2',data)
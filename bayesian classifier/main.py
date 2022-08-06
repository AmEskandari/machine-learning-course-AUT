import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

from tqdm import tqdm
from seaborn import heatmap
from scipy.stats import multivariate_normal


import os 


def train_test_split(data, test_size=0.3):

    data = data.sample(frac=1).copy()

    train = data.iloc[:int(data.shape[0]*(1-test_size)),:]
    test = data.iloc[int(data.shape[0]*(1-test_size)):,:]
    

    return train, test

#----------------------Reading The data and pre-processing -----------------------

data = pd.read_csv(os.getcwd()+'/Frogs_MFCCs.csv').drop(['Genus', 'Species', 'RecordID'], axis=1).sample(frac=1)

train,test = train_test_split(data, test_size=0.3)

x_train, y_train = train.iloc[:,:-1], train.iloc[:,-1]
x_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]

print(f'Number of Training Sampels : {x_train.shape[0]} | Number of Testing Sampels : {x_test.shape[0]}')
print(f'Each instance dimenstion : {x_train.shape[1]}')

#----------------------EDA-------------------

classes = ['Leptodactylidae' ,'Hylidae', 'Dendrobatidae' ,'Bufonidae']
classes_dic = {0:'Leptodactylidae',1:'Hylidae',2:'Dendrobatidae',3:'Bufonidae'}
classes_dic_inv = {'Leptodactylidae':0,'Hylidae':1,'Dendrobatidae':2,'Bufonidae':3}

train_class_dist = [(item/(sum(list(y_train.value_counts()))))*100 for item in list(y_train.value_counts())]
test_class_dist = [(item/(sum(list(y_test.value_counts()))))*100 for item in list(y_test.value_counts())]

#with plt.style.context('ggplot'): 

#   plt.figure(figsize = [14,6])

#   plt.subplot(1,2,1)

#   plt.bar(classes, train_class_dist, label = 'Training classes Disturbition', color = 'r')

#   plt.legend()

#   plt.subplot(1,2,2)

#   plt.bar(classes, test_class_dist, label = 'Testing classes Disturbition' , color = 'b')

#   plt.legend()

#   plt.savefig(os.getcwd()+'/disp_classes.eps', dpi = 1200)

#   plt.show()

#----------------------A-------------------


def confusion_matrix(y_true,y_prd):
    matrix = np.zeros((np.unique(y_true).size,np.unique(y_true).size),dtype=int)
    y_true = np.array(y_true)
    y_prd = np.array(y_prd)
    
    for idx in range(y_true.size):

        matrix[int(classes_dic_inv[y_true[idx]]),int(classes_dic_inv[y_prd[idx]])] += 1
    
  
    return matrix



def part_a_b_c(part):

    accuracy_training_all = []
    accuracy_testing_all = []

    for iteration,seed in enumerate([42,52]):

        train,test = train_test_split(data, test_size=0.3)

        x_train, y_train = train.iloc[:,:-1], train.iloc[:,-1]
        x_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]

        if part == 'b' or part == 'c_v2':
            p_training = [p/(sum(list(train['Family'].value_counts()))) for p in list(train['Family'].value_counts())] 
            p_testing = [p/(sum(list(test['Family'].value_counts()))) for p in list(test['Family'].value_counts())] 

        else:
            p_training = [0.25,0.25,0.25,0.25]
            p_testing = [0.25,0.25,0.25,0.25]

        means = []
        covs = []
    
        for label in classes:
            
            means.append(np.array([np.mean(x_train.loc[train['Family'] == label,col]) for col in list(x_train.columns)]))
            
            if part == 'c_v1' or part == 'c_v2':



                covs.append((np.cov(x_train.loc[train['Family'] == label].T)*np.eye(means[0].shape[0])))

            
            else:

                covs.append((np.cov(x_train.loc[train['Family'] == label].T)))
        
        

        normal_dist_Leptodactylidae = multivariate_normal(means[0],covs[0], allow_singular = True)
        normal_dist_Hylidae = multivariate_normal(means[1],covs[1], allow_singular = True)
        normal_dist_Dendrobatidae = multivariate_normal(means[2],covs[2], allow_singular = True)
        normal_dist_Bufonidae = multivariate_normal(means[3],covs[3], allow_singular = True)


        pred_traning = []
        correct_training = 0
        for idx in tqdm(range(x_train.shape[0]), desc = 'Training Phase'):
        
            out = [normal_dist_Leptodactylidae.logpdf(x_train.iloc[idx])+np.log(p_training[0]),
                   normal_dist_Hylidae.logpdf(x_train.iloc[idx])+np.log(p_training[1]),
                   normal_dist_Dendrobatidae.logpdf(x_train.iloc[idx])+np.log(p_training[2]),
                   normal_dist_Bufonidae.logpdf(x_train.iloc[idx])+np.log(p_training[3])]
            
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

        plt.title(f'Training data confusion matrix iteration {iteration+1}')

        heatmap(df_training, annot = True, fmt="d")

        plt.subplot(1,2,2)

        plt.title(f'Testing data confusion matrix iteration {iteration+1}')

        heatmap(df_testing, annot = True, fmt="d")

        plt.savefig(os.getcwd()+ f'/conf_matrix_iteraion_{iteration+1}_{part}.jpg', dpi = 400)

        print(' ')
        
    
        #plt.show()

    print('')
    print(f'Mean of Training Accuracy : {(sum(accuracy_training_all)/len(accuracy_training_all)):.2f} | Mean of Testing Accuracy : {(sum(accuracy_testing_all)/len(accuracy_testing_all)):.2f}')
    print('')


def part_d(part,data):
    
    print(data.shape)

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
                   normal_dist_Dendrobatidae.logpdf(x_train.iloc[idx])+np.log(p_training[2]),
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


def part_e(part):

    accuracy_training_all = []
    accuracy_testing_all = []

    for iteration,seed in enumerate([42,52]):

        train,test = train_test_split(data, test_size=0.3)

        x_train, y_train = train.iloc[:,:-1], train.iloc[:,-1]
        x_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]

        if part == 'v2':
            p_training = [p/(sum(list(train['Family'].value_counts()))) for p in list(train['Family'].value_counts())] 
            p_testing = [p/(sum(list(test['Family'].value_counts()))) for p in list(test['Family'].value_counts())] 

        elif part == 'v1':
            p_training = [0.25,0.25,0.25,0.25]
            p_testing = [0.25,0.25,0.25,0.25]

        means = []
        covs = []
    
        for label in classes:
            
            means.append(np.array([np.mean(x_train.loc[train['Family'] == label,col]) for col in list(x_train.columns)]))

            cov = np.zeros((x_train.shape[1],x_train.shape[1]))
        
            np.fill_diagonal(cov,[np.var(x_train.loc[train['Family'] == label,col]) for col in list(x_train.columns)])
            
            covs.append(cov)
        

        normal_dist_Leptodactylidae = multivariate_normal(means[0],covs[0], allow_singular = True)
        normal_dist_Hylidae = multivariate_normal(means[1],covs[1], allow_singular = True)
        normal_dist_Dendrobatidae = multivariate_normal(means[2],covs[2], allow_singular = True)
        normal_dist_Bufonidae = multivariate_normal(means[3],covs[3], allow_singular = True)


        pred_traning = []
        correct_training = 0
        for idx in tqdm(range(x_train.shape[0]), desc = 'Training Phase'):
        
            out = [normal_dist_Leptodactylidae.logpdf(x_train.iloc[idx])+np.log(p_training[0]),
                   normal_dist_Hylidae.logpdf(x_train.iloc[idx])+np.log(p_training[1]),
                   normal_dist_Dendrobatidae.logpdf(x_train.iloc[idx])+np.log(p_training[2]),
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

        plt.title(f'Training data confusion matrix iteration {iteration+1}')

        heatmap(df_training, annot = True, fmt="d")

        plt.subplot(1,2,2)

        plt.title(f'Testing data confusion matrix iteration {iteration+1}')

        heatmap(df_testing, annot = True, fmt="d")

        plt.savefig(os.getcwd()+ f'/part_e_conf_matrix_iteraion_{iteration+1}_{part}.jpg', dpi = 400)

        print(' ')
        print(' ')
    
        #plt.show()

    print('')
    print(f'Mean of Training Accuracy : {(sum(accuracy_training_all)/len(accuracy_training_all)):.2f} | Mean of Testing Accuracy : {(sum(accuracy_testing_all)/len(accuracy_testing_all)):.2f}')





if __name__ == '__main__':

    
    print('-----------------------------------------')
    print('Part A')
    part_a_b_c('a')

    print('-----------------------------------------')
    print('Part B')
    part_a_b_c('b')


    print('-----------------------------------------')
    print('Part C version 1: equal prior assumtion')
    part_a_b_c('c_v1')

    print('-----------------------------------------')
    print('Part C version 2: not equal prior assumtion')
    part_a_b_c('c_v2')

    print('-----------------------------------------')
    print('Part D version 1: equal prior assumtion')
    part_d('v1',data)

    print('-----------------------------------------')
    print('Part D version 2: not equal prior assumtion')
    part_d('v2',data)

    print('-----------------------------------------')
    print('Part E version 1: equal prior assumtion')
    print('Part E Version 1')
    part_e('v1')


    print('-----------------------------------------')
    print('Part E version 2: not equal prior assumtion')
    print('Part E Version 2')
    part_e('v2')

 




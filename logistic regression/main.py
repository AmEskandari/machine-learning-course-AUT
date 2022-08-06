import os
import timeit
import torch
import pandas as pd
 

import matplotlib.pyplot as plt

from src import Data, GuassianNaiveBayse, LogesticRegressionClf
from tqdm import tqdm

DATA_DIRE = os.getcwd() + "/Dataset.csv"
SEED = 99123138
TRAIN_RATIO = 0.7
VALID_RATIO = 0.1
LEARNING_RATE = 0.01
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



dataset_obj = Data(DATA_DIRE, SEED)
dataset_obj.encode_labels()
train_obj, valid_obj, test_obj = dataset_obj.splite_data(TRAIN_RATIO,VALID_RATIO)

logistic_reg_clf = LogesticRegressionClf(dataset_obj.get_num_classes(),
                                           dataset_obj.get_num_dim_features())

generate = torch.Generator()

""" Part1-a: Finding Best Number of Epohcs and Learnini_rate
    Grid Search: Epohcs Number search space = [5,25,50,100]
                 Learning Rate search space = [0.01,0.001] """


EPOCH_NUM_SPACE = [5,25,50,100]
LEARNING_RATE_SPACE = [0.01,0.005,0.001]

EPOCH_NUM_SPACE = [250,500,1000]
LEARNING_RATE_SPACE = [0.01]

with open('grid_search_res.txt', 'a') as f:
    f.write('The Grid Search Result: \n')

#Grid search
for epohc_num in EPOCH_NUM_SPACE:
    for lr in LEARNING_RATE_SPACE:
        
        generate.manual_seed(SEED)
        
        logistic_reg_clf.reset_parameters(generate)     
        logistic_reg_clf.fit(train_obj, epohc_num, lr, DEVICE, batch_size = 32)
        acc = logistic_reg_clf.evalute(valid_obj, DEVICE)

        with open('grid_search_res.txt', 'a') as f:
            f.write(f"The setting: epoch number : {epohc_num} | learning rate : {lr} | valid accuracy : {acc:.2f} \n")

generate.manual_seed(SEED)
logistic_reg_clf.reset_parameters(generate)     

dict_result = {'Epoch':[], 'Train Accuracy': [], 'Test Accuracy' : []}

for epoch in tqdm(range(100)):    
    logistic_reg_clf.fit(train_obj, 1, 0.01, DEVICE, batch_size = 32)
    acc_train =  logistic_reg_clf.evalute(train_obj, DEVICE)
    acc_test = logistic_reg_clf.evalute(test_obj, DEVICE)
    print(acc_train,acc_test)
    dict_result['Epoch'].append(epoch+1)
    dict_result['Train Accuracy'].append(acc_train)
    dict_result['Test Accuracy'].append(acc_test)

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
plt.figure(figsize=[14,7])
plt.title('Compre the Train and Test Accuracy')
plt.plot(dict_result['Epoch'],dict_result['Train Accuracy'], label = 'Train Accuracy',
        linewidth = 2)
plt.plot(dict_result['Epoch'],dict_result['Test Accuracy'], label = 'Test Accuracy',
        linewidth = 2)
plt.xlabel('Epochs', fontdict=font)
plt.ylabel('Accuracy', fontdict=font)
plt.legend()
#plt.savefig(os.getcwd()+'/result_1_a.csv.eps')





""" Part1-b: The guasssian naive bayse algorithm. 

"""

start = timeit.timeit()
gnb_clf = GuassianNaiveBayse(train_obj)
end = timeit.timeit()
elapsed_training_time = end - start

_,train_acc = gnb_clf.evalute(train_obj)
_,valid_acc = gnb_clf.evalute(valid_obj)
_,test_acc = gnb_clf.evalute(test_obj)
with open(os.path.join(os.getcwd(),'results/1','result_b.txt'),'a') as f:
    f.write(f'Training time: {elapsed_training_time} \n')
    f.write(f"""Train Accuracy: {train_acc:.2f} | Valid Accuracy: {valid_acc:.2f} 
    Test Accuracy: {test_acc:.2f}""")

logistic_reg_clf.fit(train_obj, 100, 0.01, DEVICE, batch_size = 32)
acc_train =  logistic_reg_clf.evalute(train_obj, DEVICE)
acc_test = logistic_reg_clf.evalute(test_obj, DEVICE)

with open(os.path.join(os.getcwd(),'results/1','result_b.txt'),'a') as f:
    f.write(f'--------------Logistic Regression---------------------- \n')
    f.write(f"""Train Accuracy: {acc_train:.2f} | Test Accuracy: {acc_test:.2f}""")

"""Part1-c: The regularize Logistic Regression"""


for gama in [5,1,0.1]:
    
    generate.manual_seed(SEED)
        
    logistic_reg_clf.reset_parameters(generate)     
    logistic_reg_clf.fit(train_obj, 100, 0.01, DEVICE,gama=gama, batch_size = 32)
    acc_train =  logistic_reg_clf.evalute(train_obj, DEVICE)
    acc_test = logistic_reg_clf.evalute(test_obj, DEVICE)
    acc_valid = logistic_reg_clf.evalute(valid_obj, DEVICE)
    
    with open(os.path.join(os.getcwd(),'results/1','result_c.txt'),'a') as f:
        f.write(f"""Gamma: {gama} | Train Accuracy: {acc_train:.2f} 
        Valid Accuracy: {acc_valid:.2f} | Test Accuracy : {acc_test:.2f}""") 


"""Part1-d: The result by training samples number"""



result_dict_lr = {'Train Ratio': [], 'Train Accuracy': [], 'Test Accuracy': []}
result_dict_gnb = {'Train Ratio': [], 'Train Accuracy': [], 'Test Accuracy': []}

for train_ratio in [0.25,0.375,0.50,0.625,0.75,0.875]:

    train_obj, valid_obj, test_obj = dataset_obj.splite_data(train_ratio,VALID_RATIO, per_calss=1)

    generate.manual_seed(SEED)

    logistic_reg_clf.reset_parameters(generate)  
    logistic_reg_clf.fit(train_obj, 50, 0.01, DEVICE, batch_size=32)
    
    result_dict_lr['Train Ratio'].append(train_ratio*100)
    result_dict_lr['Train Accuracy'].append(logistic_reg_clf.evalute(train_obj, DEVICE))
    result_dict_lr['Test Accuracy'].append(logistic_reg_clf.evalute(test_obj, DEVICE))

    
    gnb_clf = GuassianNaiveBayse(train_obj)

    result_dict_gnb['Train Ratio'].append(train_ratio*100)    
    result_dict_gnb['Train Accuracy'].append(gnb_clf.evalute(train_obj)[1])
    result_dict_gnb['Test Accuracy'].append(gnb_clf.evalute(test_obj)[1])

plt.figure(figsize=[18,7])

plt.subplot(1,2,1)
plt.title('Train and Test accuracy of Logistice Regression', fontdict = {'family': 'serif',
                                                        'color':  'darkred',
                                                         'weight': 'normal',
                                                             'size': 12,
                                                                         })
plt.plot(result_dict_lr['Train Ratio'], result_dict_lr['Train Accuracy'], label = 'Train Accuracy',
        linewidth = 2)
plt.plot(result_dict_lr['Train Ratio'], result_dict_lr['Test Accuracy'], label = 'Train Accuracy',
        linewidth = 2)

plt.xlabel('The Ratio of Training Examples', fontdict =  {'family': 'serif',
                                'color':  'darkred',
                                'weight': 'normal',
                                'size': 10,
                                    })
plt.ylabel('The Accuracy', fontdict =  {'family': 'serif',
                                'color':  'darkred',
                                'weight': 'normal',
                                'size': 10,
                                    })
plt.ylim([0,100])
plt.legend()

plt.subplot(1,2,2)

plt.title('Train and Test accuracy of GNB', fontdict = {'family': 'serif',
                                                        'color':  'darkred',
                                                         'weight': 'normal',
                                                             'size': 12,
                                                                         })
plt.plot(result_dict_gnb['Train Ratio'], result_dict_gnb['Train Accuracy'], label = 'Train Accuracy',
        linewidth = 2)
plt.plot(result_dict_gnb['Train Ratio'], result_dict_gnb['Test Accuracy'], label = 'Train Accuracy',
        linewidth = 2)

plt.xlabel('The Ratio of Training Examples',fontdict =  {'family': 'serif',
                                'color':  'darkred',
                                'weight': 'normal',
                                'size': 10,
                                    })
plt.ylabel('The Accuracy',fontdict =  {'family': 'serif',
                                'color':  'darkred',
                                'weight': 'normal',
                                'size': 10,
                                    })
plt.legend()
plt.ylim([0,100])

plt.savefig(os.path.join(os.getcwd(),'results/1','result_d.eps'))


"""Part2-a: """


for iteration in range(3):

    generate.manual_seed(SEED+iteration)
    dataset_obj.reset_fold()
    
    for fold in range(3):
        train_obj, test_obj = dataset_obj.get_fold(3)

        logistic_reg_clf.reset_parameters(generate)     
        logistic_reg_clf.fit(train_obj, 50, 0.01, DEVICE, batch_size = 32)
        
        train_acc = logistic_reg_clf.evalute(train_obj, DEVICE)
        test_acc = logistic_reg_clf.evalute(test_obj, DEVICE)

        with open(os.path.join(os.getcwd(),'results/2','a_result.txt'),'a') as f:
            
            f.write(f"""Iteration : {iteration} | Fold: {fold}\n
            Train Accuracy: {train_acc:.2f} | Test Accuracy : {test_acc:.2f}\n""")

"""Part2-b:  """        

for iteration in range(3):

    generate.manual_seed(SEED+iteration)
    dataset_obj.reset_fold(3)
    
    for fold in range(3):
        train_obj, test_obj = dataset_obj.get_fold(3)

        logistic_reg_clf.reset_parameters(generate)     
        logistic_reg_clf.fit(train_obj, 50, 0.01, DEVICE,gama=0.01, batch_size = 32)
        
        train_acc = logistic_reg_clf.evalute(train_obj, DEVICE)
        test_acc = logistic_reg_clf.evalute(test_obj, DEVICE)

        with open(os.path.join(os.getcwd(),'results/2','b_result.txt'),'a') as f:
            
            f.write(f"""Iteration : {iteration} | Fold: {fold}\n
            Train Accuracy: {train_acc:.2f} | Test Accuracy : {test_acc:.2f}\n""")



""" Bounos: Bounos part of assignment """


""" BN logistic regression """


logistic_reg_clf.fit_nb(train_obj)
acc_train =  logistic_reg_clf.evalute(train_obj, DEVICE)
acc_test = logistic_reg_clf.evalute(test_obj, DEVICE)
acc_valid = logistic_reg_clf.evalute(valid_obj, DEVICE)

with open(os.path.join(os.getcwd(),'results/bonous','result_reg_NB.txt'),'a') as f:
    f.write(f"""Train Accuracy: {acc_train:.2f} 
    Valid Accuracy: {acc_valid:.2f} | Test Accuracy : {acc_test:.2f}""") 


""" Different regularization term """



dic_res = {'Gamma':[], 'Validation Accuracy':[]}

for gama in [10,5,1,0.1,0.01,0.001]:

    
    generate.manual_seed(SEED)
    logistic_reg_clf.reset_parameters(generate)     
    logistic_reg_clf.fit(train_obj, 100, 0.01, DEVICE,gama=gama, batch_size = 32)
    acc_train =  logistic_reg_clf.evalute(train_obj, DEVICE)
    acc_test = logistic_reg_clf.evalute(test_obj, DEVICE)
    acc_valid = logistic_reg_clf.evalute(valid_obj, DEVICE)

    dic_res['Gamma'].append(str(gama))
    dic_res['Validation Accuracy'].append(acc_valid)

    with open(os.path.join(os.getcwd(),'results/bonous','result_reg_values.txt'),'a') as f:
        f.write(f"Gamma: {gama} | Train Accuracy: {acc_train:.2f} | Valid Accuracy: {acc_valid:.2f} | Test Accuracy : {acc_test:.2f}\n") 


#plt.figure(figsize=[14,7])
#plt.title('The Validation Accuracy vs regularization term scale')
#plt.plot(dic_res['Gamma'], dic_res['Validation Accuracy'], label = 'Validaton Accuracy',linewidth = 2)
#plt.ylim([0,100])
#plt.ylabel('Validation Accuracy')
#plt.xlabel('Regularization Term scale')
#plt.legend()
#plt.savefig(os.path.join(os.getcwd(),'results/bonous','result_reg_values.eps'))
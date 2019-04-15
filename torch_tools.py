import numpy as np
import torch
from torch import optim 
import random
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import torch.optim as optim
import time 
from sklearn.metrics import confusion_matrix
import itertools
import sklearn 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def get_key(val,my_dict): 
    """
    Simple Function to Get Key 
    in Dictionary from val. 
    
    Input: Key, Dictionary 
    Output: Val
    
    """
    for key, value in my_dict.items(): 
         if val == value: 
            return key 
    return "key doesn't exist"

def one_hot(c,classes):
    """
    Simple one hot encoding for the 
    types of arrthymia conditions. 
    
    class --> encode class
    'N' --> [1, 0, 0, 0, 0, 0, 0, 0]
    
    c:: current class of the object
    classes:: classes dictionary 
        
    """
    enc=np.zeros(len(classes),dtype=int).tolist()
    enc[get_key(c,classes)]= 1
    return enc

def get_train_test(X,y,train_size,patients):
    """
    Isolate Data Into Train and Test SubGroups 
    train_size is the number of patients that will be 
    trained upon.
    
    """
    print('Training w {} patients, Testing on {}\n'.format(train_size,(48-train_size)))
  
    train_patients= random.sample(patients, train_size)
    train_samples=[np.argwhere(y[:,0] == int(train)).flatten().tolist() for train in train_patients]
    train_samples=[item for sublist in train_samples for item in sublist]
    test_patients= np.setdiff1d(patients,train_patients)
    test_samples= [np.argwhere(y[:,0] == int(test)).flatten().tolist() for test in test_patients]
    test_samples=[item for sublist in test_samples for item in sublist]
    y_enc=np.array([get_key(c,classes) for c in y[:,2]])

    return X[train_samples],y_enc[train_samples],X[test_samples],y_enc[test_samples]

def train_val_loader(trainset,valid_size=.1,batch_size=512): # Split train into train + validation 
    """
    Get trainloader and validloader for model training

    """
    print ('Getting Data... {}% Validation Set\n'.format(int(np.around(valid_size*100))))
    
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    print('Train Len=',len(train_idx),', Validation Len=',len(valid_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                             sampler=train_sampler, num_workers=2)
    validloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                             sampler=valid_sampler,num_workers=2)
    print('Train Size Batched=',len(trainloader),', Validation Size Batched=',len(validloader))
    data_loader = {"train": trainloader, "val": validloader}
    return data_loader



#TEST DATA

def get_test_loader(test_set,batch_size=512):
    """
    Get testloader to load test data
    """

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                           shuffle=False, num_workers=2)
    print('\nTest Len=',len(test_set))
    print ('Test Size Batched=', len(test_loader))
    return test_loader

class Anomaly_Classifier(nn.Module):
    def __init__(self, input_size,num_classes):
        super(Anomaly_Classifier, self).__init__()
    
        self.conv= nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5,stride=1)
        
        self.conv_pad = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,stride=1,padding=2)
       
        self.drop_50 = nn.Dropout(p=0.5)
        self.drop_10 = nn.Dropout(p=0.1)
        self.maxpool = nn.MaxPool1d(kernel_size=5,stride=2) 

        self.dense1 = nn.Linear(32 * 8, 32) 
        self.dense2 = nn.Linear(32, 32) 
        self.dense_final = nn.Linear(32, num_classes)
        self.softmax= nn.LogSoftmax(dim=1)

    def forward(self, x):
        residual= self.conv(x)
      
        #block1 
        x = F.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x+= residual 
        x = F.relu(x)
        residual = self.maxpool(x) #[512 32 90]
       
        #block2
        x=F.relu(self.conv_pad(residual))
        x=self.conv_pad(x)
        x+=residual
        x= F.relu(x)
        residual = self.maxpool(x) #[512 32 43]
        
        #block3
        x=F.relu(self.conv_pad(residual))
        x=self.conv_pad(x)
        x+=residual
        x= F.relu(x)
        residual = self.maxpool(x) #[512 32 20]
        
        #block4
        x=F.relu(self.conv_pad(residual))
        x=self.conv_pad(x)
        x+=residual
        x= F.relu(x)
        x = self.maxpool(x) #[512 32 8]

        #MLP
        x = x.view(-1, 32 * 8) #Reshape (current_dim, 32*2)
        x = F.relu(self.dense1(x))
        x= self.drop_50(x)
        x = self.dense2(x)
        x = self.softmax(self.dense_final(x))
        return x


def reset_weights(model):
  """
  model.apply(reset_weights) will reset all the model parameters.
  This way the model is not overwhelmed 
  
  """
  if isinstance(model, nn.Conv1d) or isinstance(model, nn.Linear):
      model.reset_parameters()

def train_model(data_loader, model, n_epochs=100,print_every=10,verbose=True,plot_results=True,validation=True):
  
  """
  Model Training Function.
  Input:
    
    Dataloader: {'train':trainloader,'val':validloader} --> If no validation is used set Validation = False & dataloader= {'train':trainloader}
    model: model.cuda() if gpu will be used, else cpu
    print_every: print every n epochs 
    verbose: print out results per epoch 
    plot_results: plot the train and valid loss 
    validation: is validation set in dataloader
  
  Output:
  
    trained classifier 
  
  """

  losses=[]
  start= time.time()
  print('Training for {} epochs...\n'.format(n_epochs))
  for epoch in range(n_epochs):
      if epoch % print_every== 0:
        print('\n\nEpoch {}/{}:'.format(epoch, n_epochs - 1))
        
      
      if validation == True: 
        evaluation=['train', 'val']
      else:
        
        evaluation=['train']
        
      # Each epoch has a training and validation phase
      for phase in evaluation:
          if phase == 'train': 
              anom_classifier.train(True)  # Set model to training mode
          else:
              anom_classifier.train(False)  # Set model to evaluate mode

          running_loss = 0.0

          # Iterate over data.
          for data in data_loader[phase]:

              # get the input images and their corresponding labels
              HB, labels = data 
              HB, labels = HB.float().unsqueeze(1).cuda(), labels.squeeze(1).cuda()


              # forward + backward + optimize
              outputs = anom_classifier(HB)

              loss = criterion(outputs, labels)#loss function 

              # zero the parameter (weight) gradients
              optimizer.zero_grad()

              # backward + optimize only if in training phase
              if phase == 'train':
                  loss.backward()
                  # update the weights
                  optimizer.step()

              # print loss statistics
              running_loss += loss.item()

          losses.append(running_loss) 

          if verbose == True and epoch % print_every== 0: 
            
            print('{} loss: {:.4f} |'.format(phase, running_loss), end=' ')
          
  print('\nFinished Training  | Time:{}'.format(time.time()-start))
  if plot_results == True:
    plt.figure(figsize=(10,10))
    plt.plot(losses[0::2],label='train_loss')
    if validation == True:
      plt.plot(losses[1::2],label='validation_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.draw()
  
  return model 

def evaluate(testloader, trained_model,verbose= True):
  """
  Evaluation Metric Platfrom. Feed in the trained model 
  and test loader data. 
  
  Returns classification metric along with 
  predictions,truths
  
  """
  print('\nEvaluating....')
  truth=[]
  preds=[]
  for i, data in enumerate(testloader, 0): 
      # get the inputs
      inputs, labels = data  
      inputs, labels = inputs.cuda(), labels.squeeze(1).cuda()
      outputs = trained_model(inputs.float().unsqueeze(1))
      _, predicted = torch.max(outputs, 1)
      preds.append(predicted.cpu().numpy().tolist())
      truth.append(labels.cpu().numpy().tolist())
  preds_flat = [item for sublist in preds for item in sublist]
  truth_flat = [item for sublist in truth for item in sublist] 


  if verbose == True:
    print("TEST ACC:",accuracy_score(truth_flat,preds_flat))
    print(classification_report(truth_flat,preds_flat))

    from collections import Counter 
    print('\nTruth',sorted(Counter(truth_flat).items()))
    print('Preds',sorted(Counter(preds_flat).items()))
  
  return preds_flat,truth_flat


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
def get_kernel_size(n_h,k_h,n_w,k_w,p_h=0,s_h=1,p_w=0,s_w=1):
    """
    Kernel Measuring Function 
    """
    return [int((n_h-k_h+p_h+s_h)/s_h),int((n_w-k_w+p_w+s_w)/s_w)]    



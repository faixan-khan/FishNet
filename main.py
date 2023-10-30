import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader ,SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

from PIL import Image
import os
import numpy as np
import pandas as pd
import time
import argparse
import timm
import wandb 
import pdb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import torch
random_seed = 1234 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

# Create an argument parser object
parser = argparse.ArgumentParser(description='A simple argument parser example.')

# Add an argument to the parser
parser.add_argument('--data_root', type=str, help='parent directory where data is stored', default='Images/')
parser.add_argument('--csv_file_train', type=str, help='csv file for train data',default='anns/train.csv')
parser.add_argument('--csv_file_val', type=str, help='csv file for val data',default='anns/test.csv')
parser.add_argument('--num_workers', type=int, help='number of dataloader workers',default=6)
parser.add_argument('--label_column', type=str, help='the model will be trained to predict this feature',default="Family")
parser.add_argument('--batch_size', type=int, help='batch size',default=256)
parser.add_argument('--epochs', type=int, help='No. of epochs',default=100)
parser.add_argument('--model', type=int, help='pretrained model to train', default=4)
parser.add_argument('--wandb', type=str, help='logging to wandb', default='False')
parser.add_argument('--learning_rate', type=float, help='learning_rate',default=3e-4)
parser.add_argument('--threshold', type=float, help='learning_rate',default=0.5)
parser.add_argument('--use_two_fcs', type=bool, help='Whether to use two fcs',default=True)
parser.add_argument('--finetune', type=bool, help='Whether to finetune backbone network',default=True)
parser.add_argument('--use_pretrained', action='store_false', help='Whether to pretrained backbone network')
parser.add_argument('--use_focal', action='store_true', help='Whether to use focal loss')
parser.add_argument('--loss_type', type=str, help='loss type, choose from [CE,FL,CB,FLCB]',default='CE')
parser.add_argument('--use_log_freq', type=str, help='use log frequency to sample data', default='False')
args = parser.parse_args()

print('args', args)

'''
model mappings
0 : vit_base
1 : vit_small
2 : vit_large
3 : r34
4 : r50
5 : r101
6 : r152
7 : beit
8 : convnext
'''

map_models = ['vit_base_patch16_224', 'vit_small_patch16_224','vit_large_patch16_224','resnet34','resnet50','resnet101','resnet152', 'beit_base_patch16_224','convnext_large']

'''wand stuff'''

if args.wandb == 'True':
    wandb.init(
        # set the wandb project where this run will be logged
        project="timm models",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 3e-3,
        "architecture": map_models[args.model],
        "dataset": "FISH-NET",
        "epochs": args.epochs,
        }
    )
    

csv_file_type = 'anns/train_full_meta_new.csv' # read all classes information
meta_df = pd.read_csv(csv_file_type)

if args.label_column=="Family":
    labels_common = list(set(np.asarray(meta_df.loc[meta_df['fam_info']==0]['Family_cls'])))
    labels_medium = list(set(np.asarray(meta_df.loc[meta_df['fam_info']==1]['Family_cls'])))
    labels_rare = list(set(np.asarray(meta_df.loc[meta_df['fam_info']==2]['Family_cls'])))
    labels_all = list(set(np.asarray(meta_df.loc[meta_df['fam_info']>=0]['Family_cls'])))
    all_classes = list(set(meta_df['Family'].values))
    nClass = len(all_classes)
elif args.label_column=="Order":
    labels_common = list(set(np.asarray(meta_df.loc[meta_df['ord_info']==0]['Order_cls'])))
    labels_medium = list(set(np.asarray(meta_df.loc[meta_df['ord_info']==1]['Order_cls'])))
    labels_rare = list(set(np.asarray(meta_df.loc[meta_df['ord_info']==2]['Order_cls'])))
    labels_all = list(set(np.asarray(meta_df.loc[meta_df['ord_info']>0]['Order_cls'])))
    all_classes = list(set(meta_df['Order_new'].values))
    nClass = len(all_classes)
elif args.label_column=="Troph":
    nClass = 1
elif args.label_column=="MultiCls":
    nClass = 9
    
print('Classification at {} level, with {} classes:', args.label_column, nClass)

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, label_column="Family", split='Train'):
        self.data_frame = pd.read_csv(csv_file)
        if label_column=='Troph':
            # remove items with Troph are empty 
            bool_series = pd.isnull(self.data_frame["Troph"]) 
            self.data_frame = self.data_frame[~bool_series]
            
            mu, var = self.data_frame['Troph'].mean(), self.data_frame['Troph'].std()
            print('Troph mean/variance', mu, var)
            self.data_frame['Troph'] = (self.data_frame['Troph'] - mu) / var # normalize Trophic values
        elif label_column == 'MultiCls':
            # remove items whose attibutes are empty
            self.all_columns = ['FeedingPath','Tropical','Temperate','Subtropical','Boreal','Polar','freshwater','saltwater','brackish']
            bool_series  = np.ones(len(self.data_frame),)
            for col in self.all_columns:
                bool_col = ~pd.isnull(self.data_frame[col])
            bool_series = bool_series * bool_col
            self.data_frame = self.data_frame[bool_series.astype(np.bool)]
            
        # select the ratio to train
        self.root_dir = root_dir
        self.transform = transform
        self.label_col = label_column
        self.image_col = "image"
        self.folder_col = "Folder"
        print('csv file: {} has {} item.'.format(csv_file, len(self.data_frame)))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        
        img_name = self.data_frame.iloc[idx][self.image_col]
        img_name = img_name.split('/')[-1]
        folder = self.data_frame.iloc[idx][self.folder_col]
        img_path = os.path.join(folder,img_name)
        image = Image.open(self.root_dir + img_path)
        
        if self.label_col=="Family":
            cls_name = self.data_frame.iloc[idx][self.label_col]
            label = meta_df.loc[meta_df['Family']==cls_name]['Family_cls'].values[0]
        elif self.label_col=="Order":
            cls_name = self.data_frame.iloc[idx][self.label_col]
            if '/' in cls_name:
                cls_name = cls_name.split('/')[0]
            label = meta_df.loc[meta_df['Order_new']==cls_name]['Order_cls'].values[0]
        elif self.label_col=='Troph':
            label = self.data_frame.iloc[idx][self.label_col]
#             label = all_classes.index(cls_name)
        elif self.label_col=='MultiCls':
            label = []
            for col in self.all_columns:
                val = self.data_frame.iloc[idx][col]
                if col == 'FeedingPath':
                    if val =='pelagic':
                        val = 1
                    elif val =='benthic':
                        val = 0
                label.append(val)
            label = np.asarray(label)
        if self.transform:
            image = self.transform(image)
        return (image, label,self.root_dir + img_path)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


dataset_train = ImageDataset(csv_file=args.csv_file_train, root_dir=args.data_root, transform=transform, label_column=args.label_column)
dataset_val = ImageDataset(csv_file=args.csv_file_val, root_dir=args.data_root, transform=transform, label_column=args.label_column)


train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                    drop_last=False, num_workers=args.num_workers, pin_memory=True)
validation_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                    drop_last=False, num_workers=args.num_workers, pin_memory=True)


final_layers = {'resnet34':512, 'resnet50':2048, 'resnet101':2048, 'resnet152':2048, 'vgg16':4096, 'vit_base_patch16_224': 768, 'vit_small_patch16_224': 384, 'vit_large_patch16_224': 1024, 'beit_base_patch16_224': 768}


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

if args.loss_type in ['CB', 'FLCB']:
    all_gts = []
    for i, (img,labels,_) in enumerate(tqdm(train_loader)):
        all_gts.append(labels)
    all_gts = np.concatenate(all_gts)
    unique, samples_per_class = np.unique(all_gts, return_counts=True)
    if args.use_log_freq:
        samples_per_class = np.log(samples_per_class)
        
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
class Classifiers(nn.Module):
    def __init__(self, backbone, finetune=False, use_two_fcs=False):
        super().__init__()

        modules = list(backbone.children())[:-1]
        self.model = nn.Sequential(*modules)       
        for param in self.model.parameters():
            param.requires_grad = True if finetune else False
            
        if use_two_fcs:
            if not args.label_column =='Troph':
                self.linear = nn.Sequential(
                    nn.Linear(final_layers[map_models[args.model]], 512),
                    nn.Dropout(0.5),
                    nn.Linear(512, nClass)
                )
            else: #Trophic level regression
                self.linear = nn.Sequential(
                    nn.Linear(final_layers[map_models[args.model]], 64),
                    nn.Linear(64, nClass)
                )
        else:
            self.linear = nn.Sequential(nn.Linear(final_layers[map_models[args.model]], nClass))
        
        
    def forward(self, inputs):
        out = self.model(inputs).squeeze()
        if len(out.size())==3:
            out = out.mean(1)
        return self.linear(out)
        
    
def validate(val_loader, model, criterion):
    print('VALIDATING', flush=True)

    model.eval()
    all_preds = []
    all_gts = []
    for i, (img,labels,_) in enumerate(tqdm(val_loader)):
        inputs = img.cuda()
        target1 = labels.cuda().long()
        with torch.no_grad():
            outputs  = model(inputs)
        outputs = torch.max(outputs,-1)[1]
        all_preds.append(outputs.data.cpu().numpy())
        all_gts.append(target1.data.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_gts)

    matrix = confusion_matrix(y_true, y_pred)
    cls_acc = matrix.diagonal()/ matrix.sum(axis=1)
    cls_f1 = f1_score(y_true, y_pred, average=None)

    avg_cls_acc_1 = np.mean(cls_acc[labels_common])
    avg_cls_acc_2 = np.mean(cls_acc[labels_medium])
    avg_cls_acc_3 = np.mean(cls_acc[labels_rare])
    avg_cls_acc_4 = np.mean(cls_acc[labels_all])
    
    avg_cls_f1_1 = np.mean(cls_f1[labels_common])
    avg_cls_f1_2 = np.mean(cls_f1[labels_medium])
    avg_cls_f1_3 = np.mean(cls_f1[labels_rare])
    avg_cls_f1_4 = np.mean(cls_f1[labels_all])
    
    print('Validation accuracy:', avg_cls_acc_1, avg_cls_acc_2, avg_cls_acc_3, avg_cls_acc_4)
    print('Validation F1 score:', avg_cls_f1_1, avg_cls_f1_2, avg_cls_f1_3, avg_cls_f1_4)
    
    model.train()
    
    return avg_cls_acc_4


def validate_multi(val_loader, model, criterion):

    print('VALIDATING', flush=True)

    model.eval()
    all_preds = []
    all_gts = []
    for i, (img,labels,_) in enumerate(tqdm(val_loader)):
        inputs = img.cuda()
        target1 = labels.cuda().long()
        with torch.no_grad():
            outputs  = model(inputs)
        outputs = (outputs>args.threshold)
        all_preds.append(outputs.data.cpu().numpy())
        all_gts.append(target1.data.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_gts)
    
    f1_macro = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    print('f1 score, acc', f1_macro, acc)

    return f1_macro


def validate_reg(val_loader, model, criterion):
    print('VALIDATING', flush=True)

    model.eval()
    all_preds = []
    all_gts = []
    mu, var = 3.3477825024356997, 0.5992086371560631
    for i, (img,labels,_) in enumerate(tqdm(val_loader)):
        inputs = img.cuda()
        target1 = labels.cuda()
        with torch.no_grad():
            outputs  = model(inputs)
        all_preds.append(outputs.data.cpu().numpy())
        all_gts.append(target1.data.cpu().numpy())
    
    y_pred = np.concatenate(all_preds) * var + mu
    y_true = np.concatenate(all_gts) * var + mu

    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    
    loss = np.mean((np.concatenate(all_preds) - np.concatenate(all_gts))**2)
    print('Validation mae,rmse, loss w/o normalization:', mae, rmse, loss)
    
    model.train()
    
    return -mae # make it the larger the better

# pdb.set_trace()
print('Creating Model....', map_models[args.model])
model = timm.create_model(map_models[args.model], pretrained=args.use_pretrained, num_classes=nClass).cuda()
if not ('beit' in map_models[args.model] or 'convnext' in map_models[args.model]):
    print('do not use two FCs')
    model = Classifiers(model, finetune=args.finetune, use_two_fcs=args.use_two_fcs).cuda()
    optimizer = optim.Adam([
                            {'params': model.model.parameters(), 'lr': args.learning_rate * 0.1},
                            {'params': model.linear.parameters(), 'lr': args.learning_rate}
                        ])
else:
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

if args.label_column =='Troph':
    criterion = nn.MSELoss()
elif args.label_column =='Family' or args.label_column =='Order':
    
    from balanced_loss import Loss
    
    if args.loss_type=='FL':
        focal_loss = Loss(loss_type="focal_loss",fl_gamma=2)
        criterion = focal_loss.cuda()
    elif args.loss_type=='CB':
        ce_loss = Loss(
            loss_type="cross_entropy",
            samples_per_class=samples_per_class,
            class_balanced=True
        )
        criterion = ce_loss.cuda()
    elif args.loss_type=='FLCB':
        # class-balanced focal loss
        focal_loss = Loss(
            loss_type="focal_loss",
            beta=0.999, # class-balanced loss beta
            fl_gamma=2, # focal loss gamma
            samples_per_class=samples_per_class,
            class_balanced=True
        )
        criterion = focal_loss.cuda()
    else: # default CE loss
        criterion = nn.CrossEntropyLoss()
elif args.label_column =='MultiCls':
    criterion = nn.BCEWithLogitsLoss()
    
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    
#train model
best_score = -99999
delta = float(0.0000001)
print('Starting Training...')
for epoch in range(args.epochs):
    for batch_idx, (img, labels,_) in enumerate(tqdm(train_loader)):

        inputs = img.cuda()
        target1 = labels.cuda()
        if args.label_column =='Troph':
            target1 = target1.float()
        else:
            target1 = target1.long()
        
        output1 = model(inputs)
        loss = criterion(output1, target1)
        
        if not batch_idx %50:
            print('Epoch : %d , i : %d    loss : %0.3f'%(epoch, batch_idx, loss.data.cpu().item()),flush=True)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
        
    print(f'End of epoch: {epoch}  loss : {loss.cpu().item()} ',flush=True)
    if( epoch%5 == 0) or (epoch>30 and epoch%2 == 0):
        if args.label_column =='Troph':
            avg_score = validate_reg(validation_loader,model,criterion)
        elif args.label_column=="MultiCls":
            avg_score = validate_multi(validation_loader,model,criterion)
        else:
            avg_score = validate(validation_loader,model,criterion)

        if avg_score > best_score:
            if args.loss_type!='CE':
                ckpt_name = map_models[args.model] + '_' + str(args.label_column) + str(args.loss_type) + '.pt'
            else:
                ckpt_name = map_models[args.model] + '_' + str(args.label_column) + '.pt'
            torch.save(model.state_dict(), ckpt_name)
            best_score = avg_score
            print('best score', best_score)
        if args.wandb == 'True':
            wandb.log({"acc": avg_score, "lr": scheduler.get_lr()[0]})
if args.wandb == 'True':
    wandb.finish()

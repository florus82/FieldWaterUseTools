import sys

origin = '/workspace/'
sys.path.append('/media/')

from tqdm import tqdm
from torch.amp import autocast, GradScaler
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from FieldWaterUseTools.FuncBox.tfcl.models.ptavit3d.ptavit3d_dn import ptavit3d_dn       
from FieldWaterUseTools.FuncBox.tfcl.nn.loss.ftnmt_loss import ftnmt_loss               
from FieldWaterUseTools.FuncBox.tfcl.utils.classification_metric import Classification  
from FieldWaterUseTools.FuncBox.FieldFuncis import *



# set the rocksdb on which training will be performed
db_name = 'IACS_dilated_border_RGB_NDVI_exclude_True_without_overlap'
print(f'learn with {db_name}')

# create output dictionary
keys = ['Epoch', 'Iteration','Loss', 'Mode']
vals = [list() for _ in range(len(keys))]
res_loss  = dict(zip(keys, vals))

keys = ['Epoch', 'MCC']
vals = [list() for _ in range(len(keys))]
res_mcc  = dict(zip(keys, vals))

def train(args):
    # dummy variable to keep track of mcc
    conti = [1,2]
    mcc_dum = 0
    num_epochs = args.epochs
    batch_size = args.batch_size

    torch.manual_seed(0)
    local_rank = 0
    # torch.cuda.set_device(local_rank)
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    NClasses = 1
    nf = 96
    verbose = True
    model_config = {'in_channels': 4,
                    'spatial_size_init': (128, 128),
                    'depths': [2, 2, 5, 2],
                    'nfilters_init': nf,
                    'nheads_start': nf // 4,
                    'NClasses': NClasses,
                    'verbose': verbose,
                    'segm_act': 'sigmoid'}

    model = ptavit3d_dn(**model_config).to(local_rank)
    criterion = ftnmt_loss()
    criterionV = ftnmt_loss()
    criterion_features = ftnmt_loss(axis=[-3, -2, -1])
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, eps=1.e-6)
    scaler = GradScaler()


    train_dataset = AI4BDataset(path_to_data=f"{origin}fields/Fine_tune_dilate_True/", mode='train')
    # train_dataset = AI4BPatchDataset(path_to_data=f"{origin}fields/Fine_tune_dilate_True/", patch_size=128, stride=64, mode='train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=3, pin_memory=True)

    valid_dataset = AI4BDataset(path_to_data=f"{origin}fields/Fine_tune_dilate_True/", mode='valid')
    # valid_dataset = AI4BPatchDataset(path_to_data=f"{origin}fields/Fine_tune_dilate_True/", patch_size=128, stride=64, mode='valid')
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=3, pin_memory=True)

    start = datetime.now()
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)


    for epoch in epoch_pbar:
        tot_loss = 0
        model.train() # train function from ptavit3d_dn(torch.nn.Module) is called
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", position=1, leave=False)
        for i, data in enumerate(train_pbar):

            images, labels = data
            images = images.to(local_rank, non_blocking=True)
            labels = labels.to(local_rank, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                preds_target = model(images)
                loss = mtsk_loss(preds_target, labels, criterion, NClasses)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tot_loss += loss.item()
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

               # for export
            res_loss['Epoch'].append(epoch)
            res_loss['Iteration'].append(i)
            res_loss['Loss'].append(loss.item())
            res_loss['Mode'].append('Train')

        kwargs = monitor_epoch(model, epoch, valid_loader, res=res_loss, criterion=criterionV, NClasses=NClasses)
        kwargs['tot_train_loss'] = tot_loss
        # for export
        res_mcc['Epoch'].append(epoch)
        res_mcc['MCC'].append(kwargs['mcc'])

        # check if mcc higher than ever observed
        if kwargs['mcc'] > mcc_dum:
            mcc_dum = kwargs['mcc']
            conti[0] = model.state_dict()
            conti[1] = epoch
        
     
        #res.append 
        if verbose:
            output_str = ', '.join(f'{k}:: {v}, |===|, ' for k, v in kwargs.items())
            epoch_pbar.write(output_str)


    if verbose:
        print("Training completed in: " + str(datetime.now() - start))

    
    torch.save(conti[0], f'{origin}fields/output/models/model_state_{db_name}_{conti[1]}_CONTROL.pth') # 

    df  = pd.DataFrame(data = res_loss)
    df.to_csv(f'{origin}fields/output/loss/loss_{db_name}_{conti[1]}.csv', sep=',',index=False)

    df  = pd.DataFrame(data = res_mcc)
    df.to_csv(f'{origin}fields/output/loss/MCC_{db_name}_{conti[1]}.csv', sep=',',index=False)


def main():
    class Args:
        def __init__(self):
            self.epochs = 15
            self.batch_size = 3 # H100 test - 94GB GPU memory

    args = Args()
        
    train(args)

if __name__ == '__main__':
    main()
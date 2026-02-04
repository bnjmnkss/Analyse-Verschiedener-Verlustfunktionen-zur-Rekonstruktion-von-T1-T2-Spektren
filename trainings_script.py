import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import random
import argparse
import sys
import os
import loss_functions

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def load_data(data_folder, train_samples_per_comp=1000, val_split=0.1, test_samples_per_comp=1000):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading data to: {device}')

    # lists for all inputs and references
    all_train_inputs = []
    all_train_refs = []

    # lists for specific compartment testing
    x_comp_list = []
    y_comp_list = []

    # filenames in data folder
    filenames = [
        'training_data_nc=1.pkl',
        'training_data_nc=2.pkl',
        'training_data_nc=3.pkl',
        'training_data_nc=4.pkl',
    ]

    # loop over files
    for filename in filenames:

        path = data_folder + filename

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            inputs_raw = data['input']
            refs_raw = data['ref']

            train_inputs_raw = []
            train_refs_raw = []

            test_inputs_raw = []
            test_refs_raw = []

            # get train samples
            if train_samples_per_comp > 0:
                # set max number of train samples
                train_samples_per_comp = min(train_samples_per_comp, len(inputs_raw))

                # get first train_samples_per_comp samples from the beginning of the file
                train_inputs_raw = inputs_raw[:train_samples_per_comp]
                train_refs_raw = refs_raw[:train_samples_per_comp]

            # get test comp samples
            if test_samples_per_comp < len(inputs_raw) and test_samples_per_comp > 0:

                # check for overlapping samples
                if train_samples_per_comp + test_samples_per_comp > len(inputs_raw):
                    print(f'Not enough samples for training and testing: There are {train_samples_per_comp+test_samples_per_comp - len(inputs_raw)} overlapping samples')

                # get last test_samples_per_comp samples from the end of the file
                test_inputs_raw = inputs_raw[-test_samples_per_comp:]
                test_refs_raw = refs_raw[-test_samples_per_comp:]

                # convert test samples in float tensor and collect all
                x_comp_list.append(torch.from_numpy(test_inputs_raw).float().to(device))
                y_comp_list.append(torch.from_numpy(test_refs_raw).float().to(device))

            if len(train_inputs_raw) > 0:
                # collect all train data (not converted to tensor yet bc of train val split)
                all_train_inputs.append(train_inputs_raw)
                all_train_refs.append(train_refs_raw)

            print(f'Loaded {len(train_inputs_raw)} Train Samples and {len(test_inputs_raw)} Test Samples from {path}')

        except FileNotFoundError:
            print(f'File with path {path} not found')
            continue

    if len(all_train_inputs) > 0:

        # [[], [], []] --> [ , , , ]
        x_total = np.concatenate(all_train_inputs, axis=0)
        y_total = np.concatenate(all_train_refs, axis=0)

        print(f'Total samples loaded for training: {len(x_total)}')

        # train val split
        x_train, x_val, y_train, y_val = train_test_split(x_total, y_total, test_size=val_split, random_state=42, shuffle=True)

        # convert to float tensor
        x_train = torch.from_numpy(x_train).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)

        x_val = torch.from_numpy(x_val).float().to(device)
        y_val = torch.from_numpy(y_val).float().to(device)

        print(f'Final Train Set: {len(x_train)} Samples')
        print(f'Final Validation Set: {len(x_val)} Samples')
    
    else:
        print('No Training Samples, returning empty tensors')
        x_train = torch.empty(0).to(device)
        y_train = torch.empty(0).to(device)
        x_val = torch.empty(0).to(device)
        y_val = torch.empty(0).to(device)

    return x_train, y_train, x_val, y_val, x_comp_list, y_comp_list


def add_gaussian_noise(inputs, std=0.005, device='cuda'):

    if std > 0:
        noise = torch.randn_like(inputs, device=device) * std
        return inputs + noise
    
    return inputs


class Net(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(Net, self).__init__()

        input_layer = 64
        l1 = 512
        l2 = 512
        l3 = 512
        l4 = 1024
        l5 = 1024
        output_layer = 3600

        self.layers = nn.Sequential(

            nn.Linear(input_layer, l1),
            nn.BatchNorm1d(l1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),

            nn.Linear(l1, l2),
            nn.BatchNorm1d(l2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),

            nn.Linear(l2, l3),
            nn.BatchNorm1d(l3),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),

            nn.Linear(l3, l4),
            nn.BatchNorm1d(l4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),

            nn.Linear(l4, l5),
            nn.BatchNorm1d(l5),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),

            nn.Linear(l5, output_layer)

        )

        self._init_weights()

    # set weight initialization
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
                # set bias to 0 instead of random 
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


    def forward(self, x):
        return self.layers(x)


def train(model, x_train, y_train, x_val, y_val, x_comp_list, y_comp_list, batch_size, n_epochs, learning_rate, loss_function, noise_std=0.0, log_interval=500, comp_interval=1000, test_comp_samples=False):

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on: {device}')

    model.to(device)

    # optimization not for every gpu
    # torch.set_float32_matmul_precision('high')

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    steps_per_epoch = (len(x_train) + batch_size - 1) // batch_size
   
    # set loss
    criterion = loss_function
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'comp_losses': []
    }

    n_train_samples = len(x_train)
    n_val_samples = len(x_val)

    start_time = time.time()
    last_log_time = start_time
    print('-'*80)

    # start train loop
    for epoch in range(1, n_epochs + 1):

        # - training -

        model.train()

        train_loss_sum = 0.0

        indices = torch.randperm(n_train_samples, device=device)

        for i in range(0, n_train_samples, batch_size):

            batch_idx = indices[i:i+batch_size]

            inputs = x_train[batch_idx]
            targets = y_train[batch_idx]

            if noise_std > 0:
                inputs = add_gaussian_noise(inputs, std=noise_std, device=device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / steps_per_epoch


        # - validating -

        model.eval()

        with torch.no_grad():

            outputs = model(x_val)
            loss = criterion(outputs, y_val)
            avg_val_loss = loss.item()

        # save losses
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # log
        if epoch % log_interval == 0 or epoch == 1:

            duration = time.time() - last_log_time
            last_log_time = time.time()

            print(f'Epoch {epoch:4d}/{n_epochs} | Train Loss: {avg_train_loss:.2e} | Val Loss: {avg_val_loss:.2e} | Time: {duration:4.0f}s') 

            print('-'*80)

        # test performance on different numbers of compartments
        if test_comp_samples and (epoch % comp_interval == 0 or epoch == n_epochs):
            
            comp_losses = []

            with torch.no_grad():

                for i in range(len(x_comp_list)):

                    inputs = x_comp_list[i]
                    targets = y_comp_list[i]

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    comp_losses.append(loss.item())

            history['comp_losses'].append(comp_losses)

            comp_loss_str = '\nIndividual Compartment Losses: \n'
            for idx, c_loss in enumerate(comp_losses):
                comp_loss_str += f'  C{idx+1}: {c_loss:.2e}\n'
            print(comp_loss_str)

            print('-'*80)
    
    print(f'Training done in {(time.time() - start_time):4.0f}')

    return model, history

def save_history(path, history):

    with open(path, 'wb') as f:
        pickle.dump(history, f)
    print(f'Saved history to {path}')

def get_loss_function_by_name(args):

    loss_name = args.loss.lower()

    match loss_name:
        case 'mse':
            return loss_functions_kass.MyMSELoss()

        case 'mae':
            return loss_functions_kass.MyMAELoss()

        case 'kldiv':
            return loss_functions_kass.MyKLDivLoss(eps=1e-7)

        case 'jsd':
            return loss_functions_kass.MyJSDLoss(eps=1e-7)

        case 'wbce':
            return loss_functions_kass.MyWeightedBCELoss(beta=args.beta, eps=1e-7)

        case 'focal':
            return loss_functions_kass.MyFocalLoss(alpha=args.alpha, gamma=args.gamma, eps=1e-7)

        case 'softdice':
            return loss_functions_kass.MySoftDiceLoss(threshold=0.0005, eps=1e-7)

        case 'tversky':
            return loss_functions_kass.MyTverskyLoss(alpha=args.alpha, beta=args.beta, threshold=0.0005, eps=1e-7)

        case 'wasserstein' | 'semd':
            return loss_functions_kass.MySlicedEMDLoss(side_length=60, n_projections=100)

        case 'maetversky':
            return loss_functions_kass.MAETverskyLoss(mae_weight=2000, tversky_weight=1.0, threshold=0.0005, alpha=args.alpha, beta=args.beta, eps=1e-7)

        case 'jsdwasserstein' | 'jsdsemd':
            return loss_functions_kass.JSDWassersteinLoss(jsd_weight=620.0, wasserstein_weight=1.0, eps=1e-7, side_length=60, n_projections=50)

        case 'wbcedice':
            return loss_functions_kass.WBCEDiceLoss(wbce_weight=1.0, dice_weight=150.0, threshold=0.0005, beta=args.beta, eps=1e-7)
        
        case _:
            raise ValueError(f'Unknown Loss Function: {loss_name}. Available Loss Functions: mse, mae, wbce, focal, softdice, tversky, wassersten/semd, maetversky, jsdwasserstein/jsdsemd, wbcedice')


if __name__ == "__main__":

    # parser
    parser = argparse.ArgumentParser(description='Loss Function Training Script')

    parser.add_argument('--loss', type=str, required=True, help='Loss Function Name')

    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for Tversky/Focal')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta for Tversky/WBCE')
    parser.add_argument('--gamma', type=float, default=2.0, help='Gamma for Focal Loss')
    
    parser.add_argument('--lr_list', type=str, default="0.01,0.005,0.001,0.0005,0.0001", help='List of learning rates (comma seperated)')
    parser.add_argument('--epochs', type=int, default=5000, help='Epochs')
    parser.add_argument('--samples', type=int, default=100000, help='Samples per compartment number')
    
    parser.add_argument('--data_dir', type=str, default='/path/to/data/', help='path to data')
    parser.add_argument('--output_dir', type=str, default='/path/to/output/', help='save path')

    args = parser.parse_args()


    # train params
    batch_size_train = 1024
    dropout_prob = 0.2
    noise_std = 0.001

    # load data
    start_time_data = time.time()
    x_train, y_train, x_val, y_val, x_comp_list, y_comp_list = load_data(
        data_folder=args.data_dir, 
        train_samples_per_comp=args.samples, 
        val_split=0.1, 
        test_samples_per_comp=100
    )
    print(f'Loading data done in {(time.time() - start_time_data):4.0f}s')

    learning_rates = [float(lr) for lr in args.lr_list.split(',')]

    for lr in learning_rates:
        print(f'Training {args.loss} with learning rate = {lr}')

        set_seed(42)

        # set model
        model = Net(dropout_prob=dropout_prob)

        # set loss
        loss_function = get_loss_function_by_name(args)

        # save directory
        loss_folder = args.output_dir + args.loss + '/'
        os.makedirs(loss_folder, exist_ok=True)

        MODEL_PATH = f'{loss_folder}model.pth'
        HISTORY_PATH = f'{loss_folder}history.pth'

        # train
        trained_model, history = train(
            model=model, 
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_comp_list=x_comp_list,
            y_comp_list=y_comp_list,
            batch_size=batch_size_train,
            n_epochs=args.epochs,
            learning_rate=lr,
            loss_function=loss_function,
            noise_std=noise_std,
            log_interval=100,
            comp_interval=500,
            test_comp_samples=True
        )

        timestamp = time.strftime(f'%Y_%m_%d_%H_%M_%S')

        param_str = ""
        if args.loss == 'tversky':
            param_str = f"_a{args.alpha}_b{args.beta}"
        elif args.loss == 'focal':
            param_str = f"_g{args.gamma}"
        elif args.loss == 'wbce':
            param_str = f"_b{args.beta}"

        final_model_name = f"{MODEL_PATH[:-4]}_{timestamp}_lr_{lr}{param_str}.pth"
        final_hist_name = f"{HISTORY_PATH[:-4]}_{timestamp}_lr_{lr}{param_str}.pkl"

        # save trained model
        torch.save(trained_model.state_dict(), final_model_name)

        # save history to file
        save_history(final_hist_name, history)



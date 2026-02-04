import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import argparse
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from loss_functions_kass import (
    MyMSELoss, MyMAELoss, MyKLDivLoss, MyJSDLoss, MyWeightedBCELoss, 
    MyFocalLoss, MySoftDiceLoss, MyTverskyLoss, MySlicedEMDLoss, 
    MAETverskyLoss, JSDWassersteinLoss, WBCEDiceLoss
)

BASE_DIR = './models/'

LOSS_NAMES = ['mse', 'mae', 'kldiv', 'jsd', 'wbce', 'focal', 'softdice', 'tversky', 'semd', 'jsdsemd', 'maetversky', 'wbcedice']

# automatically generate model and history paths
BEST_MODELS = {
    name: {
        'model': f"{BASE_DIR}{name}/{name}_best_model.pth",
        'history': f"{BASE_DIR}{name}/{name}_history.pkl"
    } for name in LOSS_NAMES
}


# loaded model has to have same architecture
# we dont need weight init anymore bc we load saved weights
class Net(nn.Module):
    def __init__(self, dropout_prob=0.5, last_layer=None):
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

    def forward(self, x):
        return self.layers(x)


def load_data(data_folder, train_samples_per_comp=1000, val_split=0.1, test_samples_per_comp=1000):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading data to: {device}')

    # lists for all inputs and references
    all_train_inputs = []
    all_train_refs = []

    # lists for specific compartment testing
    x_comp_list = []
    y_comp_list = []

    t1_t2_grid = None

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

            if t1_t2_grid is None and 't1_t2_combinations' in data:
                t1_t2_grid = data['t1_t2_combinations']
                print(f'Grid loaded successfully: {t1_t2_grid.shape}')

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

        print(f'Total Train Samples loaded: {len(x_total)}')

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
        print('No training samples, returning empty tensors')
        x_train = torch.empty(0).to(device)
        y_train = torch.empty(0).to(device)
        x_val = torch.empty(0).to(device)
        y_val = torch.empty(0).to(device)

    return x_train, y_train, x_val, y_val, x_comp_list, y_comp_list, t1_t2_grid


def calculate_metrics(y_target, y_pred):

    # get max value for range scaling
    vmax = y_target.max()
    if vmax == 0: vmax = 1.0 # fallback

    # mse
    mse = ((y_target - y_pred)**2).mean()

    # dice score
    threshold = 0.0005
    mask_target = y_target > threshold
    mask_pred = y_pred > threshold

    intersection = np.logical_and(mask_target, mask_pred).sum()
    sum_masks = mask_target.sum() + mask_pred.sum()

    if sum_masks == 0:
        dice = 0.0
    else:
        dice = (2. * intersection) / sum_masks

    # ssim (structural similarity)
    ssim = structural_similarity(y_target, y_pred, data_range=vmax)

    # psnr (peak signal to noise ratio)
    psnr = peak_signal_noise_ratio(y_target, y_pred, data_range=vmax)

    return mse, dice, ssim, psnr


def viz_sample_2d(y_target, y_pred, grid, title='Prediction vs. Ground Truth [2D]'):

    if grid is not None:
        unique_t1 = np.unique(grid[:, 0])
        unique_t2 = np.unique(grid[:, 1])
        n_t1, n_t2 = len(unique_t1), len(unique_t2)

        step_x = 10
        step_y = 10

        xticks_pos = np.arange(0, n_t2, step_x)
        xticks_lab = [f'{int(v)}' for v in unique_t2[::step_x]]

        yticks_pos = np.arange(0, n_t1, step_y)
        yticks_lab = [f'{int(v)}' for v in unique_t1[::step_y]]

        xlabel = 'T2 [ms]'
        ylabel = 'T1 [ms]'

    else:
        xticks_pos, xticks_lab = None, None
        yticks_pos, yticks_lab = None, None
        xlabel, ylabel = 'Index X', 'Index Y'

    # calculate difference between prediction and target
    diff = np.abs(y_target - y_pred)

    # calculate evaluation metrics
    mse, dice, ssim, psnr = calculate_metrics(y_target, y_pred)

    # get max value for scaling
    vmax = max(y_target.max(), y_pred.max())
    if vmax == 0: vmax = 1.0 # fallback

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(f'{title} \nMSE: {mse:.2e} | Dice: {dice:.2f} | SSIM: {ssim:.2f} | PSNR: {psnr:.2f}', fontsize=14)

    def set_axis_labels(ax):
        if grid is not None:
            ax.set_xticks(xticks_pos)
            ax.set_xticklabels(xticks_lab)
            ax.set_yticks(yticks_pos)
            ax.set_yticklabels(yticks_lab)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(y_target, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
    ax1.set_title('Ground Truth')
    set_axis_labels(ax1)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(y_pred, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
    ax2.set_title('Prediction')
    set_axis_labels(ax2)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(diff, cmap='inferno', origin='lower')
    ax3.set_title('Difference')
    set_axis_labels(ax3)

    plt.tight_layout()
    plt.show()

def viz_sample_3d(y_target, y_pred, grid, title='Prediction vs. Ground Truth [3D]'):

    # calculate difference between prediction and target
    diff = np.abs(y_target - y_pred)

    if grid is not None:
        unique_t1 = np.unique(grid[:, 0])
        unique_t2 = np.unique(grid[:, 1])
        
        X, Y = np.meshgrid(unique_t2, unique_t1)
        xlabel, ylabel = 'T2 [ms]', 'T1 [ms]'
    else:
        x = np.arange(0, 60, 1)
        y = np.arange(0, 60, 1)
        X, Y = np.meshgrid(x, y)
        xlabel, ylabel = 'Index X', 'Index Y'

    # calculate evaluation metrics
    mse, dice, ssim, psnr = calculate_metrics(y_target, y_pred)

    # get max value for scaling
    vmax = max(y_target.max(), y_pred.max())
    if vmax == 0: vmax = 1.0 # fallback

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f'{title} \nMSE: {mse:.2e} | Dice: {dice:.2f} | SSIM: {ssim:.2f} | PSNR: {psnr:.2f}', fontsize=14)

    def plot_surf(ax, Z, t, cmap):
        ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False, alpha=0.9)
        ax.set_title(t)
        ax.set_zlim(0, vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.view_init(elev=30, azim=-60)

    plot_surf(fig.add_subplot(1, 3, 1, projection='3d'), y_target, 'Ground Truth', 'viridis')
    plot_surf(fig.add_subplot(1, 3, 2, projection='3d'), y_pred, 'Prediction', 'viridis')
    plot_surf(fig.add_subplot(1, 3, 3, projection='3d'), diff, 'Difference', 'inferno')

    plt.tight_layout()
    plt.show()

def plot_history(history, model_name, skip_epochs=5, log_scale=False):

    train_loss = history['train_loss']
    val_loss = history['val_loss']

    if len(train_loss) <= skip_epochs:
        print('skip_epochs is larger that total epochs: showing all epochs')
        skip_epochs = 0

    epochs = range(1 + skip_epochs, len(train_loss) + 1)

    train_loss = train_loss[skip_epochs:]
    val_loss = val_loss[skip_epochs:]

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_loss, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_loss, label='Val Loss', linewidth=2,)

    plt.title(f'Train vs. Val Loss (Epochs {1+skip_epochs} - {len(train_loss)+skip_epochs}) for {model_name}', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', ls='-', alpha=0.2)

    if log_scale:
        plt.yscale('log')
        plt.ylabel('Loss (Log Scale)', fontsize=12)

    if not log_scale:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.show()


def plot_overview_grid(model, x_comp_list, y_comp_list, grid, m_name):
    rows, cols = 4, 2 
    fig, axes = plt.subplots(rows, cols, figsize=(6.5, 11), constrained_layout=True)
    
    col_headers = ["Ground Truth", "Prediction"]
    row_labels = ["1 Comp", "2 Comp", "3 Comp", "4 Comp"]

    for i in range(rows):
        if len(x_comp_list[i]) == 0: continue
        idx = np.random.randint(len(x_comp_list[i]))
        
        x_in = x_comp_list[i][idx].unsqueeze(0)

        if  m_name in ['softdice', 'tversky']:
            # if softdice or tversky: hard binarize target
            tensor_data = y_comp_list[i][idx].reshape(60, 60)
            y_target = (tensor_data > 0.0005).float().cpu().numpy()
        else:
            y_target = y_comp_list[i][idx].cpu().numpy().reshape(60, 60)
        
        model.eval()
        with torch.no_grad():
            logits = model(x_in)
            if  m_name in ['softdice', 'tversky']:
                # if softdice or tversky: apply sigmoid, else softmax
                y_pred = torch.sigmoid(logits).cpu().numpy().reshape(60, 60)
            else:
                y_pred = F.softmax(logits, dim=-1).cpu().numpy().reshape(60, 60)

        vmax = max(y_target.max(), y_pred.max())
        
        axes[i, 0].imshow(y_target, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
        axes[i, 1].imshow(y_pred, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
        
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])

        axes[i, 0].text(-0.2, 0.5, row_labels[i], transform=axes[i, 0].transAxes, 
                        va='center', ha='right', fontsize=10, fontweight='bold', rotation=90)

    for ax, col_name in zip(axes[0], col_headers):
        ax.set_title(col_name, fontsize=10, pad=12, fontweight='bold')

    plt.show()


def evaluate_quantitative(model, x_comp_list, y_comp_list, m_name, batch_size=1024):
    print("\n" + "="*61)
    print(f"{'Comp':<6} | {'Samples':<9} | {'MSE':<10} | {'Dice':<8} | {'SSIM':<8} | {'PSNR':<8}")
    print("-" * 61)

    for i, (x_data, y_data) in enumerate(zip(x_comp_list, y_comp_list)):
        n_comp, n_samples = i + 1, len(x_data)
        if n_samples == 0: continue
            
        sum_mse = sum_dice = sum_ssim = sum_psnr = 0.0
        
        with torch.no_grad():
            for b in range(0, n_samples, batch_size):
                x_batch = x_data[b:b+batch_size]
                y_batch = y_data[b:b+batch_size]

                logits = model(x_batch)

                if m_name in ['softdice', 'tversky']:
                    # if softdice or tversky use sigmoid as activation + hard binarize targets
                    preds = torch.sigmoid(logits).cpu().numpy()
                    y_batch_np = (y_batch > 0.0005).float().cpu().numpy()
                else:
                    preds = F.softmax(logits, dim=-1).cpu().numpy()
                    y_batch_np = y_batch.cpu().numpy()
                
                for k in range(len(y_batch_np)):
                    m, d, s, p = calculate_metrics(y_batch_np[k].reshape(60,60), preds[k].reshape(60,60))
                    sum_mse += m; sum_dice += d; sum_ssim += s; sum_psnr += p

        print(f"{n_comp:<6} | {n_samples:<9} | {sum_mse/n_samples:.2e}   | {sum_dice/n_samples:.4f}   | {sum_ssim/n_samples:.4f}   | {sum_psnr/n_samples:.2f}")
    print("="*61 + "\n")


def evaluate_loss_matrix(models_dict, x_comp_list, y_comp_list, batch_size=1024):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    loss_criterions = {
        'mse': MyMSELoss().to(device),
        'mae': MyMAELoss().to(device),
        'kldiv': MyKLDivLoss(eps=1e-7).to(device),
        'jsd': MyJSDLoss(eps=1e-7).to(device),
        'wbce': MyWeightedBCELoss(beta=10.0, eps=1e-7).to(device),
        'focal': MyFocalLoss(alpha=0.25, gamma=2.0, eps=1e-7).to(device),
        'softdice': MySoftDiceLoss(threshold=0.0005, eps=1e-7).to(device),
        'tversky': MyTverskyLoss(threshold=0.0005, alpha=0.9, beta=0.9, eps=1e-7).to(device),
        'semd': MySlicedEMDLoss(side_length=60, n_projections=50).to(device),
        'jsdsemd': JSDWassersteinLoss(jsd_weight=620.0, wasserstein_weight=1.0, eps=1e-7, side_length=60, n_projections=50).to(device),
        'maetversky': MAETverskyLoss(mae_weight=2000.0, threshold=0.0005, tversky_weight=1.0, alpha=0.1, beta=0.9, eps=1e-7).to(device),
        'wbcedice': WBCEDiceLoss(wbce_weight=1.0, dice_weight=150.0, beta=10.0, eps=1e-7).to(device)
    }
    
    all_metrics = list(loss_criterions.keys())
    matrix = {m_name: {metric: 0.0 for metric in all_metrics} for m_name in models_dict.keys()}

    x_total = torch.cat(x_comp_list, dim=0)
    y_total = torch.cat(y_comp_list, dim=0)
    n_samples = len(x_total)

    sigmoid_models = ['softdice', 'tversky']
    hybrid_models = ['maetversky', 'wbcedice']

    for model_name, paths in models_dict.items():
        if not os.path.exists(paths['model']): 
            print(f"Model {model_name} skipped (not found).")
            continue
        
        model = Net()
        model.load_state_dict(torch.load(paths['model'], map_location=device))
        model.to(device)
        model.eval()

        if model_name in sigmoid_models:
            current_activation = 'sigmoid'
        elif model_name in hybrid_models:
            current_activation = None
        else:
            current_activation = 'softmax'

        sum_losses = {metric: 0.0 for metric in all_metrics}
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                x_batch = x_total[i:i+batch_size]
                y_batch = y_total[i:i+batch_size]
                logits = model(x_batch)

                weight = len(x_batch) / n_samples
                
                for metric_name, criterion in loss_criterions.items():

                    # set eval mode = True for softdice, tversky, maetversky, wbcedice bc they need hard binarized input for accurate results
                    eval_mode = metric_name in ['softdice', 'tversky', 'maetversky', 'wbcedice']

                    # quick fix bc only those loss fkts have eval mode as parameter in forward method
                    if metric_name in ['softdice', 'tversky', 'maetversky', 'wbcedice']:
                        val = criterion(logits, y_batch, activation=current_activation, eval_mode=eval_mode).item()
                    else:
                        val = criterion(logits, y_batch, activation=current_activation).item()

                    sum_losses[metric_name] += val * weight

        matrix[model_name] = sum_losses

    first_col_width = 15
    col_width = 12
    total_width = first_col_width + col_width * len(all_metrics)

    print("\n" + "=" * total_width)
    print("CROSS-LOSS MATRIX")
    print("=" * total_width)
    
    header = f"{'MODELL':<{first_col_width}}" + "".join([f"{m:>{col_width}}" for m in all_metrics])
    print(header)
    print("-" * total_width)

    for model_name in matrix.keys():
        if sum(matrix[model_name].values()) == 0: continue
        
        row_str = f"{model_name.upper():<{first_col_width}}"
        for metric in all_metrics:
            val = matrix[model_name][metric]
            val_str = f"{val:.2e}" if val < 0.001 else f"{val:.4f}"
            row_str += f"{val_str:>{col_width}}"
        print(row_str)

    print("=" * total_width + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    
    parser.add_argument('--model', type=str, nargs='+', choices=list(BEST_MODELS.keys()) + ['all'], required=True, 
                        help="model(s) used for evaluation")
    
    valid_tasks = ['all', 'matrix', 'table', 'overview', 'indepth', 'history']
    parser.add_argument('--tasks', nargs='+', choices=valid_tasks, default=['table'],
                        help="evaluation task(s)")

    parser.add_argument('--data_dir', type=str, default='/path/to/data/', help='path to data')
    DATA_FOLDER = args.data_dir

    parser.add_argument('--samples', type=int, default=10000, help='Testsamples per compartment number')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'all' in args.model:
        selected_models = list(BEST_MODELS.keys())
    else:
        selected_models = args.model

    tasks_to_run = ['matrix', 'table', 'overview', 'indepth', 'history'] if 'all' in args.tasks else args.tasks

    # dont load data if task is only history
    if tasks_to_run != ['history']:
        # load data
        x_train, y_train, x_val, y_val, x_comp, y_comp, grid = load_data(
            data_folder=DATA_FOLDER, 
            train_samples_per_comp=0, 
            val_split=0.0, 
            test_samples_per_comp=args.samples
        )
        
    # show matrix
    if 'matrix' in tasks_to_run:
        models_to_eval = {m: BEST_MODELS[m] for m in selected_models}
        evaluate_loss_matrix(models_to_eval, x_comp, y_comp)


    # other individual tasks per model
    other_tasks = [t for t in tasks_to_run if t != 'matrix']
    
    if other_tasks:
        for m_name in selected_models:
            paths = BEST_MODELS[m_name]
            
            print(f"evaluating model: {m_name.upper()}")

            # plot history
            if 'history' in other_tasks:
                try:
                    with open(paths['history'], 'rb') as f:
                        history = pickle.load(f)
                    plot_history(history, m_name.upper())
                except FileNotFoundError:
                    print(f'File with path {history_path} not found')

            # if task is only history dont load model
            if other_tasks != ['history']:
                # load model
                model = Net()
                model.load_state_dict(torch.load(paths['model'], map_location=device))
                model.to(device)
                model.eval()

            # display evaluation metrics as table
            if 'table' in other_tasks:
                evaluate_quantitative(model, x_comp, y_comp, m_name)

            # ground truth vs. prediction for 1 to 4 comps
            if 'overview' in other_tasks:
                plot_overview_grid(model, x_comp, y_comp, grid, m_name)

            # 2d and 3d and difference plots displayed seperately for every comp
            if 'indepth' in other_tasks:
                for n_comp in range(1, 5):
                    idx = np.random.randint(len(x_comp[n_comp-1]))
                    x_in = x_comp[n_comp-1][idx].unsqueeze(0)
                    if  m_name in ['softdice', 'tversky']:
                        # if softdice or tversky: hard binarize target
                        tensor_data = y_comp[n_comp-1][idx].reshape(60, 60)
                        y_tar = (tensor_data > 0.0005).float().cpu().numpy()
                    else:
                        y_tar = y_comp[n_comp-1][idx].cpu().numpy().reshape(60, 60)
                    
                    with torch.no_grad():
                        logits = model(x_in)
                        if  m_name in ['softdice', 'tversky']:
                            # if softdice or tversky: apply sigmoid, else softmax
                            y_prd = torch.sigmoid(logits).cpu().numpy().reshape(60, 60)
                        else:
                            y_prd = F.softmax(logits, dim=-1).cpu().numpy().reshape(60, 60)

                    viz_sample_2d(y_tar, y_prd, grid, title=f'{m_name.upper()} | {n_comp} Comp | 2D')
                    viz_sample_3d(y_tar, y_prd, grid, title=f'{m_name.upper()} | {n_comp} Comp | 3D')

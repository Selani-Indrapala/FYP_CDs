import models.nn_processor as nn_processor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import toml
from tqdm import tqdm

from audio_dataset import AudioDataset  # Import the custom dataset class

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

# Load configuration from config.toml
config = toml.load('/kaggle/working/FYP_CDs/LA_Code/configs/BaseModel.toml')

# Extract model and training parameters
input_dim = config['model']['input_dim']
hidden_dim = config['model']['hidden_dim']
n_heads = config['model']['n_heads']
n_layers = config['model']['n_layers']
num_classes = config['model']['num_classes']
lr = config['training']['learning_rate']
epochs = config['training']['epochs']
batch_size = config['training']['batch_size']
device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

# Initialize the model using the loaded configuration
model = nn_processor.AudioDeepfakeTransformer(input_dim, n_heads, n_layers, hidden_dim, num_classes)

# Move model to device before defining optimizer
model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define the paths to the data
flac_train_folder =  '/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_train/flac' 
train_labels = '/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
flac_val_folder = '/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_dev/flac' 
val_labels = '/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

# Initialize the dataset and dataloaders
train_dataset = AudioDataset(flac_train_folder, train_labels)
val_dataset = AudioDataset(flac_val_folder, val_labels)  # For validation, you can define a separate folder for val data if needed

def custom_collate_fn(batch):
    # Separate inputs and labels from the batch of dictionaries
    inputs = [sample['features'] for sample in batch]
    labels = [sample['label'] for sample in batch]
    
    # Ensure inputs are tensors (not strings or other types)
    assert all(isinstance(input, torch.Tensor) for input in inputs), "Inputs must be tensors, not strings"
    
    # Find the maximum length in the batch (based on input size)
    max_len = max(input.size(0) for input in inputs)
    
    # Pad each input to the maximum length
    padded_inputs = torch.zeros(len(inputs), max_len, inputs[0].size(1))  # Assuming 2D tensors (e.g., [time, feature])
    for i, input in enumerate(inputs):
        padded_inputs[i, :input.size(0), :] = input  # Copy input into padded tensor
    
    # Convert labels to a tensor
    labels = torch.tensor(labels)
    
    return padded_inputs, labels

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    losses = checkpoint.get('losses', [])
    min_tDCF_best = checkpoint.get('min_tDCF_best', float('inf'))  # Default to infinity if not found
    optimal_threshold_best = checkpoint.get('optimal_threshold', None) 
    print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
    return start_epoch, losses, min_tDCF_best, optimal_threshold_best
    
# Evaluation function
def compute_min_tDCF_and_threshold(scores, labels):
    """
    Compute the minimum t-DCF and the optimal threshold for classification.
    
    Args:
        scores (list or np.array): Raw scores from the model's output (probabilities or logits).
        labels (list or np.array): True binary labels (0 or 1).
        
    Returns:
        min_tDCF (float): The minimum t-DCF value.
        optimal_threshold (float): The optimal threshold to use.
    """
    # Define the cost function weights
    P_target = 0.01  # Prior probability of target speaker
    P_non_target = 0.99  # Prior probability of non-target speaker
    C_miss = 1  # Miss detection cost (false negative)
    C_fa = 1  # False alarm cost (false positive)
    
    # Sort the scores to calculate thresholds and corresponding confusion matrix
    thresholds = np.linspace(0, 1, 1000)  # 1000 possible thresholds
    min_tDCF = float('inf')
    optimal_threshold = 0

    for threshold in thresholds:
        # Convert scores to binary predictions using the threshold
        predictions = (np.array(scores) > threshold).astype(int)

        # Calculate confusion matrix for this threshold
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

        # Compute the miss and false alarm rates
        Pmiss = fn / (fn + tp)  # Miss detection rate
        Pfa = fp / (fp + tn)  # False alarm rate
        
        # Compute t-DCF (target detection cost function)
        tDCF = P_target * C_miss * Pmiss + P_non_target * C_fa * Pfa
        
        # Track minimum t-DCF and corresponding threshold
        if tDCF < min_tDCF:
            min_tDCF = tDCF
            optimal_threshold = threshold
    
    return min_tDCF, optimal_threshold

def evaluate(model, dataloader, device):
    """
    Evaluate the model's performance on the validation set, calculating min t-DCF and optimal threshold.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): Validation data loader.
        device (torch.device): The device to run the evaluation on.

    Returns:
        float: Minimum t-DCF value.
        float: Optimal decision threshold.
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for sample in dataloader:
            inputs = sample['features']
            labels = sample['label']

            # Send to device
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.view(-1)  # Flatten the outputs to shape: [batch_size]
            
            # Collect scores and labels
            all_scores.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute min t-DCF and optimal threshold
    min_tDCF, optimal_threshold = compute_min_tDCF_and_threshold(all_scores, all_labels)

    # Print the evaluation results
    print(f"Validation Min t-DCF: {min_tDCF:.4f}, Optimal Threshold: {optimal_threshold:.4f}")
    
    return min_tDCF, optimal_threshold

# Check if checkpoint exists
checkpoint_dir = "/kaggle/working/checkpoints/FullDataset"
os.makedirs(checkpoint_dir, exist_ok=True)
latest_checkpoint = None

# Find the latest checkpoint (if any)
if os.listdir(checkpoint_dir):
    latest_checkpoint = max([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)], key=os.path.getctime)

# Load checkpoint if available
if latest_checkpoint:
    print(f"Found checkpoint: {latest_checkpoint}")
    start_epoch, losses, min_tDCF_best, optimal_threshold = load_checkpoint(model, optimizer, latest_checkpoint)
else:
    start_epoch = 0
    min_tDCF_best = float('inf')  # Initialize as infinity if no checkpoint is found
    losses = [] 

# Directory for saving loss plots
plot_dir = "/kaggle/working/checkpoints/loss_plots"
os.makedirs(plot_dir, exist_ok=True)

# Main training and evaluation loop
for epoch in range(start_epoch, epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    
    # Training phase
    model.train()
    total_loss = 0
    correct_preds = 0
    total_samples = 0
    # Initialize tqdm progress bar
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
        for sample in train_loader:
            inputs = sample['features']
            if inputs.shape[1]>8792:
                print(inputs.shape[1])
            labels = sample['label']

            # Send to device
            inputs, labels = inputs.to(device), labels.to(device).float()

            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze()  # Adjust for binary classification shape if necessary
            loss = criterion(outputs, labels.squeeze(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate predictions
            preds = (outputs > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

            total_loss += loss.item()
            #print(f'Total loss is {str(total_loss)}')
            # Update the progress bar
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update(1)  # Increment progress bar

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)  # Save average loss
    train_accuracy = correct_preds / total_samples
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")
    
    # Evaluation phase
    if (epoch + 1) % 1 == 0:
        min_tDCF, optimal_threshold = [0,0.5] #evaluate(model, val_loader, device)

        if min_tDCF <= min_tDCF_best:
            print(f'New minimum t DCF found. {str(min_tDCF_best)} -> {str(min_tDCF)}')
            min_tDCF_best = min_tDCF

            # Save best checkpoint
            checkpoint_path = f"/kaggle/working/checkpoints/FullDataset/best_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'min_tDCF_best': min_tDCF_best,  # Save the best min t-DCF value
                'optimal_threshold': optimal_threshold,  # Save the optimal threshold
            }, checkpoint_path)
            print(f"Best checkpoint saved with min t-DCF: {min_tDCF:.4f} at epoch {epoch+1}. Path: {checkpoint_path}")

        # Save checkpoint after each epoch
        checkpoint_path = f"/kaggle/working/checkpoints/FullDataset/epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'losses': losses,
            'min_tDCF_best': min_tDCF_best,
            'optimal_threshold': optimal_threshold,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    
        # Plot the training loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses, marker='o', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.legend()

        # Save the plot
        plot_path = os.path.join(plot_dir, f"loss_curve_V2.png")
        plt.savefig(plot_path)
        print(f"Loss curve saved to {plot_path}")
        plt.close()  # Close the plot to free up memory

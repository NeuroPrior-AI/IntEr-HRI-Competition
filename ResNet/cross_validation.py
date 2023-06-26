import torch
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
import torch.nn as nn
from CRNN.datasets.dataset import EEGDataset
from models.resnet import ResNet
import os
import numpy as np
from tools.standardize import standardize
from train import train_model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from predict import predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tools.f1 import calculate_f1_score


if __name__ == "__main__":
    root_directory = "C:/Users/PaulS/Desktop/IntErHRI_data/csv_epoch"
    participants = os.listdir(root_directory)

    # For logging (edit before training):
    log_name = "resnet_cycliclr_80epochs_68batch"
    log_dir = "./tmp/crossval_logs/" + log_name
    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Check gpu availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f'Using device: {device}.')

    for p in participants:
        print(f'Started with participant {p}.')

        p_dir = log_dir + f"/{p}"
        # Create the log directory if it doesn't exist
        os.makedirs(p_dir, exist_ok=True)

        conf_mats = []      # stores all confusion matrices across 10 trials
        f1s = []            # stores all f1s across 10 trials

        # Load data
        X = np.load(f'./tmp/cross_validation_data/X_{p}.npy')
        y = np.load(f'./tmp/cross_validation_data/y_{p}.npy')

        # Define 10-fold cross validation
        kfold = KFold(n_splits=10, shuffle=True, random_state=1)

        for i, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f'Started {i} trial of k-fold cross valdiation.')

            trial_dir = p_dir + f"/trial{i}"       # directory will be created in train_model()

            # Split into train and test sets
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Standardize data
            X_train, X_val = standardize(X_train, X_val)

            # Build model
            model = ResNet(num_classes=2)
            model.to(device)

            # Initialize dataloaders
            batch_size = 136
            train_dataset = EEGDataset(X_train, y_train, device)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # each set has 68 trials
            val_dataset = EEGDataset(X_val, y_val, device)
            val_dataloader = DataLoader(dataset=val_dataset)

            # Specify training hyperparameters:
            learning_rate = 0.001
            epochs = 80
            alpha = 1
            xentropy_weight = torch.tensor([
                (68 / 62) ** alpha,  # S 1 + S 32 + S 48 + S 64 + S 80
                (68 / 6) ** alpha  # S 96
            ]).to(device)
            criterion = nn.CrossEntropyLoss(weight=xentropy_weight)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, mode="triangular2", step_size_up=400,
                                 cycle_momentum=False)

            # Run the training loop:
            print(f'Start training.')
            train_accs, test_accs, train_losses, test_losses, learning_rates = train_model(model,
                                                                                           optimizer,
                                                                                           criterion,
                                                                                           None,
                                                                                           train_dataloader,
                                                                                           val_dataloader,
                                                                                           epochs=epochs,
                                                                                           print_every=8,
                                                                                           log_dir=trial_dir)

            # Save and plot graphs:
            plt.plot(train_losses)
            plt.plot(test_losses)
            plt.title("loss")
            plt.savefig(trial_dir + "/loss.png")

            plt.plot(train_accs)
            plt.plot(test_accs)
            plt.title("accuracy")
            plt.savefig(trial_dir + "/accuracy.png")

            plt.plot(learning_rates)
            plt.plot(learning_rates)
            plt.title("learning_rates")
            plt.savefig(trial_dir + "/learning_rates.png")
            print(f'Saved training plots.')

            # Save and plot confusion matrix:
            model = torch.load(trial_dir + "/best_model.pt")
            model.to(device)
            labels, preds = predict(model, val_dataloader)
            conf_mat = confusion_matrix(labels, preds, normalize="true")
            if conf_mat.shape == (1, 1):        # edge case if validation set only contains 1 class
                conf_mat = np.eye(2)        # in this case, we can be sure the model can learn a single class classifier perfectly
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
            disp.plot()
            plt.savefig(trial_dir + "/cm_best_model.png")
            plt.show()
            print(f'Saved confusion matrix.')

            conf_mats.append(conf_mat)
            f1s.append(calculate_f1_score(conf_mat))

            # Record f1 for each trial
            with open(trial_dir + "/f1.txt", "w") as f:
                f.write(f"Model f1 score: {calculate_f1_score(conf_mat)}.")
            print(f"Model f1 score: {calculate_f1_score(conf_mat)}.")

        # After all 10-fold cross validation:
        avg_f1 = np.mean(f1s)
        avg_conf_mat = np.sum(conf_mats, axis=0) / len(conf_mats)

        # Save and plot average confusion matrix:
        disp = ConfusionMatrixDisplay(confusion_matrix=avg_conf_mat)
        disp.plot()
        plt.savefig(p_dir + "/avg_cm.png")
        plt.show()
        print(f'Saved average confusion matrix.')

        # Write avg f1, avg conf mat log
        with open(p_dir + "/avg.txt", "w") as f:
            f.write(f"Average f1 score: {avg_f1}.")
            f.write(f"Average conf mat: {avg_conf_mat}.")

        print(f"Average f1 score: {avg_f1}.")
        print(f"Average conf mat: {avg_conf_mat}.")
        print("==================")


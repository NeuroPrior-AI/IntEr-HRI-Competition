import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CyclicLR
from models.resnet import ResNet
from datasets.dataset import EEGDataset
from predict import predict
from sklearn.model_selection import train_test_split
from tools.standardize import standardize
from tools.f1 import calculate_f1_score


def train_model(model, optimizer, criterion, lr_scheduler, train_loader, test_loader, epochs=100, print_every=10, log_dir="new_log"):
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    learning_rates = []
    best_val_loss = np.inf

    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    # Define the log file path
    log_file = os.path.join(log_dir, "training_log.txt")

    model.train()
    for epoch in range(epochs):
        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        for i, (inputs, labels) in enumerate(train_loader):
            model.zero_grad()
            pred = model(inputs)
            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()

            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            if lr_scheduler is not None:
                lr_scheduler.step()
                current_lr = lr_scheduler.get_last_lr()[0]

            learning_rates.append(current_lr)
            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            labels = torch.max(labels.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()

        accuracy = correct / total
        test_acc, test_loss = evaluate(model, test_loader, criterion)

        # Save model with the best validation loss so far
        if test_loss < best_val_loss:
            torch.save(model, log_dir + "/best_model.pt")
            best_val_loss = test_loss
            # Write log
            with open(log_file, "a") as f:
                f.write(f"Saved model at {epoch} epoch.\n")
            print(f"Saved model at {epoch} epoch.")

        log_epoch = "Epoch {}, Train acc: {:.2f}%, Test acc: {:.2f}%, Train loss: {:.2f}, Test loss: {:.2f}" \
            .format(epoch, accuracy * 100, test_acc * 100, xentropy_loss_avg, test_loss)
        # Write log
        with open(log_file, "a") as f:
            f.write(log_epoch + "\n")

        if epoch % print_every == 0:
            print(log_epoch)

        train_accs.append(accuracy)
        test_accs.append(test_acc)
        train_losses.append(xentropy_loss_avg / (i + 1))
        test_losses.append(test_loss)

    return train_accs, test_accs, train_losses, test_losses, learning_rates


def evaluate(model, loader, criterion):
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    val_loss = 0.
    for i, (inputs, labels) in enumerate(loader):
        with torch.no_grad():
            pred = model(inputs)
            xentropy_loss = criterion(pred, labels)
            val_loss += xentropy_loss.item()

        pred = torch.max(pred.data, 1)[1]
        labels = torch.max(labels.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()

    val_acc = correct / total
    val_loss = val_loss / (i + 1)
    model.train()
    return val_acc, val_loss


if __name__ == "__main__":
    # Check gpu availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f'Using device: {device}.')

    # Load data
    # We have 3258 training examples and 1087 test examples.
    X = np.load('./tmp/data/X.npy')
    y = np.load('./tmp/data/y.npy')
    X_val = np.load('./tmp/data/X_val.npy')
    y_val = np.load('./tmp/data/y_val.npy')
    # X = np.load('./tmp/cross_validation_data/X_AA56D.npy')
    # y = np.load('./tmp/cross_validation_data/y_AA56D.npy')
    print(f'Loaded data.')

    # Split into train/val set:
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Standardize data
    X_train, X_val = standardize(X, X_val)
    print(f'Standardized data.')

    # Build model
    model = ResNet(num_classes=2)
    model.to(device)

    # Initialize dataloaders
    batch_size = 136
    train_dataset = EEGDataset(X, y, device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # each set has 68 trials
    val_dataset = EEGDataset(X_val, y_val, device)
    val_dataloader = DataLoader(dataset=val_dataset)

    # Specify training hyperparameters:
    learning_rate = 0.001       # if test loss explode, use smaller lr
    epochs = 80
    alpha = 1
    xentropy_weight = torch.tensor([
        (68 / 62) ** alpha,  # S 1 + S 32 + S 48 + S 64 + S 80
        (68 / 6) ** alpha  # S 96
    ]).to(device)
    criterion = nn.CrossEntropyLoss(weight=xentropy_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, mode="triangular2", step_size_up=100,
                         cycle_momentum=False)

    # For logging (edit before training):
    log_model = "resnet_crossval"
    log_epochs = str(epochs) + "epochs"
    log_batch = str(batch_size) + "batch"
    log_name = log_model + "_" + log_batch + "_" + log_epochs
    log_dir = "./tmp/train_logs/" + log_name

    # Run the training loop:
    print(f'Start training.')
    train_accs, test_accs, train_losses, test_losses, learning_rates = train_model(model,
                                                                                   optimizer,
                                                                                   criterion,
                                                                                   None,
                                                                                   train_dataloader,
                                                                                   val_dataloader,
                                                                                   epochs=epochs,
                                                                                   print_every=1,
                                                                                   log_dir=log_dir)

    # Save model
    torch.save(model, log_dir + "/latest_model.pt")
    print(f'Saved model.')

    # Save and plot graphs:
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.title("loss")
    plt.savefig(log_dir + "/loss.png")
    plt.show()

    plt.plot(train_accs)
    plt.plot(test_accs)
    plt.title("accuracy")
    plt.savefig(log_dir + "/accuracy.png")
    plt.show()

    plt.plot(learning_rates)
    plt.plot(learning_rates)
    plt.title("learning_rates")
    plt.savefig(log_dir + "/learning_rates.png")
    plt.show()
    print(f'Saved training plots.')

    # Save and plot confusion matrix:
    # For latest model
    labels, preds = predict(model, val_dataloader)
    conf_mat = confusion_matrix(labels, preds, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot()
    plt.savefig(log_dir + "/cm_latest_model.png")

    # For best model
    model = torch.load(log_dir + "/best_model.pt")
    model.to(device)
    labels, preds = predict(model, val_dataloader)
    conf_mat = confusion_matrix(labels, preds, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot()
    plt.savefig(log_dir + "/cm_best_model.png")
    plt.show()
    print(f'Saved confusion matrices.')

    # Print f1
    print(f"f1: {calculate_f1_score(conf_mat)}")

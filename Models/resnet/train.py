import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CyclicLR
from resnet import ResNet
from eegnet import EEGNet
from crnn import CRNN
from dataset import EEGDataset
from sklearn.model_selection import train_test_split
from utils import calculate_f1_score, plot_confusion_matrix, predict, standardize


def train_model(model, optimizer, criterion, lr_scheduler, train_loader, val_loader, epochs, print_every=10, log_dir="new_log"):
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    learning_rates = []
    best_val_loss = np.inf

    # Define the log file path
    log_file = os.path.join(log_dir, "training_log.txt")

    for epoch in range(epochs):
        # Train model:
        model.train()
        total_loss = 0.
        total_correct = 0.
        total_samples = 0.

        for i, (inputs, labels) in enumerate(train_loader):
            model.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)     # labels should be 1D and is the class index
            loss.backward()

            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            if lr_scheduler is not None:
                lr_scheduler.step()
                current_lr = lr_scheduler.get_last_lr()[0]

            learning_rates.append(current_lr)
            total_loss += loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]       # get class indices of each prediction
            total_samples += labels.size(0)
            total_correct += (pred == labels.data).sum().item()

        # Evaluate model after batch training for an epoch:
        model.eval()
        accuracy = total_correct / total_samples
        avg_loss = total_loss / (i + 1)
        val_acc, val_loss = evaluate(model, val_loader, criterion)

        # Save model with the best validation loss so far
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), log_dir + "/best_model.pt")
            best_val_loss = val_loss
            # Write log
            with open(log_file, "a") as f:
                f.write(f"Saved model at {epoch} epoch.\n")
            print(f"Saved model at {epoch} epoch.")

        log_epoch = "Epoch {}, Train acc: {:.2f}%, Val acc: {:.2f}%, Train loss: {:.2f}, Val loss: {:.2f}" \
            .format(epoch, accuracy * 100, val_acc * 100, avg_loss, val_loss)
        # Write log
        with open(log_file, "a") as f:
            f.write(log_epoch + "\n")

        if epoch % print_every == 0:
            print(log_epoch)

        train_accs.append(accuracy)
        test_accs.append(val_acc)
        train_losses.append(avg_loss)
        test_losses.append(val_loss)

    return train_accs, test_accs, train_losses, test_losses, learning_rates


def evaluate(model, loader, criterion):
    model.eval()
    total_correct = 0.
    total_samples = 0.
    total_loss = 0.
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            pred = model(inputs)
            loss = criterion(pred, labels)
            total_loss += loss.item()

            pred = torch.max(pred.data, 1)[1]       # get class indices of each prediction
            total_samples += labels.size(0)
            total_correct += (pred == labels.data).sum().item()

    acc = total_correct / total_samples
    avg_loss = total_loss / (i + 1)

    return acc, avg_loss


if __name__ == "__main__":
    # Check gpu availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f'Using device: {device}.')

    # Load data
    all_data_path = '../../Preprocess'
    with open(all_data_path + '/X.pkl', 'rb') as f:
        X = pickle.load(f)
        # X = np.expand_dims(X, axis=1)       # for testing with EEGNet
    with open(all_data_path + '/y.pkl', 'rb') as f:
        y = pickle.load(f)
        y = y - 1       # This is only because when generating y.pkl class indices start from 1

    print(f'X shape: {X.shape}, y shape: {y.shape}')    # X should have shape (num_samples, 64, 501), y (num_samples,)

    # Split into 70% training, 10% validation, 20% testing:
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.66667, random_state=42)
    print(f'Loaded data.')
    class_0_percentage = np.sum(Y_train == 0) / len(Y_train)
    class_1_percentage = 1 - class_0_percentage
    print(f'Training class dist: {class_0_percentage:.2f} vs. {class_1_percentage:.2f}.')

    # Standardize data
    X_train, X_val, X_test, mean, std = standardize(X_train, X_val, X_test)
    print(f'Standardized data.')

    # Build model
    model = CRNN(num_classes=2)
    model.to(device)

    # Initialize dataloaders
    batch_size = 136
    train_dataset = EEGDataset(X_train, Y_train, device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # each set has 68 trials
    val_dataset = EEGDataset(X_val, Y_val, device)
    val_dataloader = DataLoader(dataset=val_dataset)
    test_dataset = EEGDataset(X_test, Y_test, device)
    test_dataloader = DataLoader(dataset=test_dataset)

    # Specify training hyperparameters:
    learning_rate = 0.00001       # if test loss explode, use smaller lr
    epochs = 80
    alpha = 0.9
    xentropy_weight = torch.tensor([
        (1 / float(class_0_percentage)) ** alpha,
        (1 / float(class_1_percentage)) ** alpha
    ]).to(device)
    criterion = nn.CrossEntropyLoss(weight=xentropy_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, mode="triangular2", step_size_up=200,
                         cycle_momentum=False)

    # For logging (edit before training):
    log_name = f'crnn_0.9alpha_hidden_size=16_bidirectional_{epochs}_epochs'
    log_dir = "../../resnet_training_logs/" + log_name
    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    # Save the mean and std of the training data of this session
    np.save(log_dir + '/mean.npy', mean)
    np.save(log_dir + '/std.npy', std)

    # Run the training loop:
    print(f'Start training.')
    train_accs, test_accs, train_losses, test_losses, learning_rates = train_model(model=model,
                                                                                   optimizer=optimizer,
                                                                                   criterion=criterion,
                                                                                   lr_scheduler=None,
                                                                                   train_loader=train_dataloader,
                                                                                   val_loader=val_dataloader,
                                                                                   epochs=epochs,
                                                                                   print_every=1,
                                                                                   log_dir=log_dir)

    # Save model
    torch.save(model.state_dict(), log_dir + "/latest_model.pt")
    print(f'Saved latest model.')

    # Save and plot training graphs:
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

    # Evaluation on test set:
    class_0_percentage = np.sum(Y_test == 0) / len(Y_test)
    class_1_percentage = 1 - class_0_percentage
    print(f'Test class dist: {class_0_percentage:.2f} vs. {class_1_percentage:.2f}.')
    test_val_file = os.path.join(log_dir, "test_eval.txt")

    # For latest model:
    # Save and plot confusion matrix:
    labels, preds = predict(model, test_dataloader)
    conf_mat = confusion_matrix(labels, preds, normalize="true")
    plot_confusion_matrix(conf_mat, 'Latest confusion matrix', log_dir + "/cm_latest_model.png")

    # Find loss and accuracy:
    test_acc, test_loss = evaluate(model, test_dataloader, criterion)
    with open(test_val_file, "a") as f:
        f.write(f"Latest model: Test loss: {test_loss}. Test acc: {test_acc}. Test f1: {calculate_f1_score(conf_mat)}.\n")
    print(f"Latest model: Test loss: {test_loss}. Test acc: {test_acc}. Test f1: {calculate_f1_score(conf_mat)}")

    # For best model:
    # Save and plot confusion matrix:
    model = CRNN(num_classes=2)
    checkpoint = torch.load(log_dir + "/best_model.pt")
    model.load_state_dict(checkpoint)
    model.to(device)
    labels, preds = predict(model, test_dataloader)
    conf_mat = confusion_matrix(labels, preds, normalize="true")
    plot_confusion_matrix(conf_mat, 'Best confusion matrix', log_dir + "/cm_best_model.png")

    # Find loss, accuracy, and f1:
    test_acc, test_loss = evaluate(model, test_dataloader, criterion)
    with open(test_val_file, "a") as f:
        f.write(f"Best model: Test loss: {test_loss}. Test acc: {test_acc}. Test f1: {calculate_f1_score(conf_mat)}.\n")
    print(f"Best model: Test loss: {test_loss}. Test acc: {test_acc}. Test f1: {calculate_f1_score(conf_mat)}")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from preprocessing.main import integrate_data
from models.crnn import CRNN
from datasets.dataset import EEGDataset
import json
from torch.optim.lr_scheduler import CyclicLR


def train_model(model, optimizer, train_loader, test_loader, lr_scheduler, epochs=100, print_every=10):
    # Using GPUs in PyTorch is pretty straightforward
    if torch.cuda.is_available():
        print("Using cuda")
        use_cuda = True
        device = torch.device("cuda")
    else:
        device = "cpu"

    xentropy_weight = torch.tensor([1 / 30 ** 1.25, 1 / 56 ** 1.25, 1 / 14 ** 1.25]).to(device) # TODO: change weights

    criterion = nn.CrossEntropyLoss(weight=xentropy_weight)
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    learning_rates = []

    # Move the model to GPU, if available
    model.to(device)
    model.train()

    for epoch in range(epochs):
        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        for i, (inputs, inputs_freq, inputs_scl, labels, lengths) in enumerate(train_loader):
            model.zero_grad()
            inputs = inputs.to(device)      # TODO: do I need this? I already put it to device when init in datasets
            labels = labels.to(device)
            # inputs = inputs.view(inputs.size(0), -1)  # Flatten input from [batch_size, 1, 28, 28] to [batch_size, 784]
            pred = model(inputs)
            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()

            optimizer.step()
            lr_scheduler.step()

            current_lr = lr_scheduler.get_last_lr()[0]
            learning_rates.append(current_lr)

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        test_acc, test_loss = evaluate(model, test_loader, criterion, device)
        if epoch % print_every == 0:
            print("Epoch {}, Train acc: {:.2f}%, Test acc: {:.2f}%".format(epoch, accuracy * 100, test_acc * 100))
            print("Epoch {}, Train loss: {:.2f}, Test loss: {:.2f}".format(epoch, xentropy_loss_avg, test_loss))

        train_accs.append(accuracy)
        test_accs.append(test_acc)
        train_losses.append(xentropy_loss_avg / i)
        test_losses.append(test_loss)

    return train_accs, test_accs, train_losses, test_losses, learning_rates


def evaluate(model, loader, criterion, device):
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    val_loss = 0.
    for i, (inputs, inputs_freq, inputs_scl, labels, lengths) in enumerate(loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            inputs_freq = inputs_freq.to(device)
            labels = labels.to(device)
            # inputs = inputs.view(inputs.size(0), -1)
            pred = model(inputs, inputs_freq, inputs_scl)
            xentropy_loss = criterion(pred, labels)
            val_loss += xentropy_loss.item()

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    val_loss = val_loss / i
    model.train()
    return val_acc, val_loss


if __name__ == "__main__":
    # Check gpu availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()
    # Load data:        # TODO: put eeg data into tensors
    validation_id_list = [135, 157, 248, 183, 214, 264]

    X1 = None
    X1f = None
    X1s = None
    t1 = None
    X2 = None
    X2f = None
    X2s = None
    t2 = None

    with open('../preprocessing/white_list.json') as json_file:
        white_list = json.load(json_file)

    for id in white_list:
        try:
            X, X_freq, X_scl, t = integrate_data(int(id), white_list[id])
            if int(id) not in validation_id_list:
                if X1 is None:
                    X1 = X
                    X1f = X_freq
                    X1s = X_scl
                    t1 = t
                else:
                    X1 = np.concatenate((X1, X), axis=0)
                    X1f = np.concatenate((X1f, X_freq), axis=0)
                    X1s = np.concatenate((X1s, X_scl), axis=0)
                    t1 = np.concatenate((t1, t), axis=1)
            else:
                if X2 is None:
                    X2 = X
                    X2f = X_freq
                    X2s = X_scl
                    t2 = t
                else:
                    X2 = np.concatenate((X2, X), axis=0)
                    X2f = np.concatenate((X2f, X_freq), axis=0)
                    X2s = np.concatenate((X2s, X_scl), axis=0)
                    t2 = np.concatenate((t2, t), axis=1)
        except:
            print(f"Something went wrong for id {id}")

    # Load/Build model
    # MODEL_PATH = "./model.pt"
    # model = torch.load(MODEL_PATH)
    model = CRNN(num_classes=3, in_channels=X.shape[1], model='lstm')

    # Initialize dataloaders
    train_dataset = EEGDataset(X1, t1, device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4096)
    # val_dataset = EEGDataset(X2, t2, device)
    # val_dataloader = DataLoader(datasets=val_dataset)

    # Visualize model
    # torch.onnx.export(model, dummy_input, "./model.onnx")
    # dummy_input = torch.randn(4096, X.shape[1], 25*30)
    # torch.onnx.export(model, dummy_input, "./model.onnx")

    # Train model:
    learning_rate = 0.01
    epochs = 240
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, mode="triangular2", step_size_up=200, cycle_momentum=False)

    # Run the training loop:
    train_accs, test_accs, train_losses, test_losses, learning_rates = train_model(model, optimizer, train_dataloader,
                                                                                   None,
                                                                                   scheduler,
                                                                                   epochs=epochs,
                                                                                   print_every=1)

    torch.save(model, "./model.pt")
    print("Model Saved")

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.title("loss")
    plt.show()

    plt.plot(train_accs)
    plt.plot(test_accs)
    plt.title("accuracy")
    plt.show()

    plt.plot(learning_rates)
    plt.plot(learning_rates)
    plt.title("learning_rates")
    plt.show()

from tqdm import tqdm

import torch
torch.manual_seed(1)
from torch import nn, Tensor

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

file = open("log/log.txt", "w+")

def train(train_loader, val_loader, model, optimizer, criterion, epoch, patience, device, model_name):
    file.write(f"\nBegin training: {model_name}")

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }
    

    last_val_epoch_loss = float('inf')
    trigger_times = 0
    for e in tqdm(range(1, epoch+1)):
        
        # training
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            if model_name == "transformer":
                X_train_batch, src_mask = transformerX_reshape(X_train_batch, device)
                y_train_pred = model(X_train_batch, src_mask)

            if model_name == "mlp":
                y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            
            
        # validation
        with torch.no_grad():
            
            val_epoch_loss = 0
            val_epoch_acc = 0
            
            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                if model_name == "transformer":
                    X_val_batch, src_mask = transformerX_reshape(X_val_batch, device)
                    y_val_pred = model(X_val_batch, src_mask)
                
                if model_name == "mlp":
                    y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()


        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))

        # for early stopping
        if val_epoch_loss > last_val_epoch_loss:
            trigger_times += 1
            file.write(f'\nTrigger Times:, {trigger_times}')

            if trigger_times >= patience:
                file.write('\nEarly stopping!\nStart to test process.')
                plot(accuracy_stats, loss_stats, model_name)
                return model

        else:
            file.write('\ntrigger times: 0')
            trigger_times = 0

        last_val_epoch_loss = val_epoch_loss


        file.write(f'\nEpoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.5f}| Val Acc: {val_epoch_acc/len(val_loader):.5f}')

    plot(accuracy_stats, loss_stats, model_name)

    return model



def test(test_loader, model, device, model_name):

    y_pred_list = []
    y_test = []
    with torch.no_grad():
        model.eval()
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            if model_name == "transformer":
                X_batch, src_mask = transformerX_reshape(X_batch, device)
                y_test_pred = model(X_batch, src_mask)

            if model_name == "mlp":
                y_test_pred = model(X_batch)

            # Convert the tensor to a numpy object and append it to the list.
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            # for each prediction
            for y_pred in y_pred_tags:
                y_pred_list.append(y_pred.cpu().numpy())
            # for each ground truth
            for y_label in y_batch:
                y_test.append(y_label.cpu().numpy())

    # for plots
    label_dic = {"AU":0, "UK": 1, "US": 2}
    label_dic_reverse = {v: k for k, v in label_dic.items()}


    # Flatten out the list so that we can use it as an input to confusion_matrix and classification_report.
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_pred_list = [label_dic_reverse[i] for i in y_pred_list]
    y_test = [a.tolist() for a in y_test]
    y_test = [label_dic_reverse[i] for i in y_test]


    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list))
    # confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)
    sns.heatmap(confusion_matrix_df, annot=True, cmap='Blues', fmt='d')
    plt.savefig(f"plot/{model_name}_confusion_matrix.pdf")
    file.write(f"\n{classification_report(y_test, y_pred_list)}")
    plt.close()


def transformerX_reshape(X_batch, device):
    X_batch = X_batch.reshape(X_batch.size(1), X_batch.size(0), X_batch.size(2))
    batch_size = X_batch.size(1)
    sequence_len = X_batch.size(0)
    src_mask = generate_square_subsequent_mask(sequence_len).to(device)
    # file.write(f"\nBatch size: {batch_size}, sequence_len: {sequence_len}")
    return X_batch, src_mask


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc



def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def plot(accuracy_stats, loss_stats, model_name):
    # Create dataframes
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

    # Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
    plt.savefig(f"plot/{model_name}_train_val_loss_acc.pdf")
    plt.close()



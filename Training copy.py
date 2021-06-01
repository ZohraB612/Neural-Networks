import numpy as np
import pandas as pd
import datetime

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, auc
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import joblib
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from Data import *
from datetime import datetime
import torch.utils.data
from CNN import CNN, count_parameters
from MLP import MLP
from sklearn.metrics import roc_curve, roc_auc_score
from pytorchtools import EarlyStopping


# Scikit-learn (2020). 'sklearn.metrics.roc_curve'. [Online].
# Available at: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
#Tarroni, G. (2021). ‘Slides provided by the lecturer at City, University of London’.

"""
this function defines the two networks and their respective configs
"""
def make_network(config):
    if config["network"] == "cnn":
        return CNN(config)
    else:
        return MLP(config)


"""
we define a function to return the torch optimisers
"""
def make_optim(name):
    if name == "sgd":
        return t.optim.SGD;
    if name == "adam":
        return t.optim.Adam

    raise Exception("Optimzer not supported " + name)


"""
this function is used to train and validate 
"""
def train_and_validate(train_loader, valid_loader, config):
    print(config)

    n_epochs = config["n_epochs"]
    learning_rate = config["lr"]
    patience = config["patience"]
    
    model = make_network(config)
    calc_loss = nn.CrossEntropyLoss()
    optimizer = make_optim(config["optimizer"])(model.parameters(), lr=learning_rate)

    print(model)
    count_parameters(model)

    losses_train = []
    losses_valid = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
            
    for epoch in range(n_epochs):
        loss_train = 0
        loss_valid = 0
        # training
        model.train()
        for data, labels in train_loader:
            # print(data.shape, labels.shape)
            optimizer.zero_grad()
            output = model(data)
            # print(labels)
            loss = calc_loss(output, labels)
            # print(loss)
            loss.backward()
            optimizer.step()

            loss_train += loss.item() * data.size(0)
            # print(epoch, ':', loss_train)
            # break

        # epoch
        model.eval()
        for data_v, labels_v in valid_loader:
            output_v = model(data_v)
            loss_v = calc_loss(output_v, labels_v)
            loss_valid += loss_v.item() * data_v.size(0)
            #break

        # early_stopping looks at the validation and uses patience
        # parameter to stop the function is validation loss increases over 
        # length of the patience value 
        early_stopping(loss_valid, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
        # average losses
        loss_train = loss_train / len(train_loader.sampler)
        loss_valid = loss_valid / len(valid_loader.sampler)

        losses_train.append(loss_train)
        losses_valid.append(loss_valid)

        print(f"Epoch {epoch}, training loss {loss_train} validation loss {loss_valid}")

    plot_results(losses_train, losses_valid)

    return model


"""
this function is used to plot the training and validation losses
"""
def plot_results(loss_train, loss_valid):
    # PLOT TRAIN/VALID
    x_axis = (range(len(loss_train)))
    plt.plot(x_axis, loss_train, 'r', label='Training loss')
    plt.plot(x_axis, loss_valid, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    
"""
this function is used to test the models
"""

# TESTING
def test_model(model, test_loader):
    model.eval()
    y_true = []
    y_predict = []
    with torch.no_grad():
        correct_predictions = 0
        total_predictions = 0
        for data, label in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total_predictions += label.size(0)
            correct_predictions += (predicted == label).sum().item()
            accuracy = (correct_predictions / total_predictions) * 100

            y_true.extend(list(label))
            y_predict.extend(list(predicted))

    print(f"Total Predictions {total_predictions} , correct predictions {correct_predictions} ")
    print(f"Accuracy of the model on test set is {accuracy}")

    # Plot confusion matrix
    plot_cm(y_true, y_predict)
    plt.show();
 
    # Determines AUC score
    auc = roc_auc_score(y_true, y_predict)
    print('AUC: %.2f' % auc)
    
    # Plots ROC 
    fpr, tpr, thresholds = roc_curve(y_true, y_predict)
    plot_roc_curve(fpr, tpr)
    plt.show();
    
    # iteration over the dataset and classification labelling
    # true and predicted
    rows = 2
    columns = 4
    classes = ('cat', 'dog')
    test_iterator = iter(test_loader)
    data, label = test_iterator.next()
    fig2 = plt.figure(figsize = (15, 8))
    for i in range(8):
        fig2.add_subplot(rows, columns, i+1)
        plt.title('truth ' + classes[label[i]] + ': predict ' + classes[predicted[i]])
        img = data[i] / 2 + 0.5     # this is to unnormalize the image
        img = torchvision.transforms.ToPILImage()(img)
        plt.imshow(img)
    plt.show()

    return accuracy

"""
function to plot the ROC curve
"""
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
"""
function to plot confusion matrix
"""
def plot_cm(y_true, y_pred):
    cm0 = confusion_matrix(y_true, y_pred)
    print(cm0)

    cm = confusion_matrix(y_true, y_pred, normalize="all")
    print(cm)

    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=["cats", "dogs"])
    display.plot()
    plt.show()


"""
train and test function using 'optuna' or single configs for the models
This is considered our objective function
"""
# method = ["single", "optuna"]
def train_and_test(trial=None, method="optuna", network="mlp"):
    if not trial is None:
        config = {
            "network": network,
            "n_epochs": 30,
            "patience": 5,
            "lr": trial.suggest_loguniform(name="lr", low=0.001, high=0.01),
            "optimizer": trial.suggest_categorical(name="optimizer", choices=["sgd", "adam"]),
            "activation": trial.suggest_categorical(name="activation", choices=["relu", "tanh", "sigmoid"]),
            "momentum": trial.suggest_loguniform(name="momentum", low=0.1, high=0.9),
            "sample": 0.1,
        }
    elif network == "mlp":
        config = {
            "network": network,
            "n_epochs": 10,
            "patience": 2,
            "lr": 0.08,
            "optimizer": "adam",
            "activation": "relu",
            "momentum": 0.9,
            "sample": 1  # 100%
        }
        # CNN
    else:
        config = {
            "network": network,
            "n_epochs_stop": 30,
            "patience": 10,
            "lr": 0.09,
            "optimizer": "adam",
            "activation": "sigmoid",
            "momentum": 0.9,
            "sample": 1  # 100%
        }
    train_loader, valid_loader, test_loader = get_loaders(config["sample"])
    print("Train: ", len(train_loader.dataset))
    print("Valid: ", len(valid_loader.dataset))

    model = train_and_validate(train_loader=train_loader,
                               valid_loader=valid_loader,
                               config=config)

    # saves model
    date_str = datetime.now().strftime("%Y_%m_%d_%H_%M")

    fp = "{0}-{1}-{2}-{3}-{4}-{5}".format(
        method,
        network,
        date_str,
        config["lr"],
        config["activation"],
        config["optimizer"].__class__.__name__)

    print("Saving the model to ", fp)
    torch.save(model.state_dict(), fp)

    return test_model(model, test_loader)

"""
this function is used to test the best models
"""

def test_best_model(config, fp):
    _, _, test_loader = get_loaders()

    model = make_network(config)
    model.load_state_dict(torch.load(fp))

    test_model(model, test_loader)


def view_study_results(network='mlp'):
    if network == 'cnn':
        study = joblib.load("cnn_study.pkl")
        df = study.trials.dataframe()
        print(df.head())
    else:
        study = joblib.load("mlp_study.pkl")
        df = study.trials.dataframe()
        print(df.head())

"""
this function is used to run optuna trials
"""

def run_optuna(network='mlp'):
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=10),
                                direction="maximize")
    unique_trials = 100
    while unique_trials > len(set(str(t.params) for t in study.trials)):
        study.optimize(train_and_test, n_trials= 1, timeout=200)
    fig = optuna.visualization.plot_optimization_history(study)
    fig1 = optuna.visualization.plot_param_importances(study)
    joblib.dump(study, f"{network}_study.pkl")
    trial = study.best_trial

    print('Accuracy {}'.format(trial.value))
    print('Best hyperparameters: {}'.format(trial.params))
    fig.show()
    fig1.show()


if __name__ == "__main__":
    pass

from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch, numpy as np
import itertools

def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """
    # raise NotImplementedError('train')

    lr = float(model_factory['lr'])
    mom = float(model_factory['mom'])
    wd = float(model_factory['wd'])
    n_epochs = int(model_factory['ep'])
    batch_size = int(model_factory['bs'])

    train = load_data('data/train', shuffle=True, batch_size=batch_size)
    valid = load_data('data/valid', shuffle=False, batch_size=batch_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)

    loss = ClassificationLoss()
    max_acc_valid = 0

    # Start training
    for epoch in range(n_epochs):

        # Iterate
        predictions, targets = [], []
        valid_predictions, valid_targets = [], []
        model.train()

        for train_batch, train_label in train:

            optimizer.zero_grad()
            # Compute the loss
            o = model(train_batch)
            loss_val = loss.forward(o, train_label)
            print(f'loss {loss_val}')

            loss_val.backward()
            optimizer.step() # Gradient becomes NaN if learning rate is too large

            predictions.append(o)
            targets.append(train_label)

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        train_accuracy = accuracy(predictions, targets)

        model.eval()
        for valid_batch, valid_label in valid:
            with torch.no_grad():
                valid_predictions.append(model(valid_batch))
                valid_targets.append(valid_label)

        valid_predictions = torch.cat(valid_predictions)
        valid_targets = torch.cat(valid_targets)
        valid_accuracy = accuracy(valid_predictions, valid_targets)

        # Evaluate the model
        if max_acc_valid < valid_accuracy:
            max_acc_valid = valid_accuracy
            save_model(model)
            print(f'train accuracy: {train_accuracy}, valid accuracy : {valid_accuracy}, num_epoch : {epoch}')


    print('\n\n')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    parser.add_argument('-l', '--lr')
    parser.add_argument('-o', '--mom')
    parser.add_argument('-w', '--wd')
    parser.add_argument('-e', '--ep')
    parser.add_argument('-b', '--bs')


    # Put custom arguments here

    args = parser.parse_args()

    model_factory['lr'] = args.lr
    model_factory['mom'] = args.mom
    model_factory['wd'] = args.wd
    model_factory['ep'] = args.ep
    model_factory['bs'] = args.bs

    train(args)

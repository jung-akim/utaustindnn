from .models import CNNClassifier, save_model, set_data_dir_and_device

from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import multiprocessing


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def train(args):
    from os import path
    model = CNNClassifier(args, device)
    set_data_dir_and_device(model, args.data_dir, device)
    model.to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """
    cpu_count = multiprocessing.cpu_count()
    train = load_data(args.data_dir + '/train', shuffle=True, batch_size=int(args.batch), num_workers=cpu_count)
    valid = load_data(args.data_dir + '/valid', shuffle=False, batch_size=int(args.batch), num_workers=cpu_count)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr = float(args.lr),
                                momentum = float(args.mom),
                                weight_decay = float(args.wd))

    loss = torch.nn.CrossEntropyLoss()
    max_acc_valid = 0

    # Start training
    counter = 0
    for epoch in range(int(args.epoch)):

        # Iterate
        predictions, targets = [], []
        valid_predictions, valid_targets = [], []
        model.train()

        for train_batch, train_label in train:

            optimizer.zero_grad()
            # Compute the loss
            train_batch = train_batch.to(device)
            train_label = train_label.to(device)

            o = model(train_batch)
            loss_val = loss.forward(o, train_label)
            # print(f'loss {loss_val}')

            loss_val.backward()
            optimizer.step() # Gradient becomes NaN if learning rate is too large

            predictions.append(o)
            targets.append(train_label)

            counter += 1
            train_logger.add_scalar('loss', loss_val, global_step=counter)

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        train_accuracy = accuracy(predictions, targets)
        train_logger.add_scalar('train_accuracy', train_accuracy, global_step=counter)

        model.eval()
        for valid_batch, valid_label in valid:
            with torch.no_grad():
                valid_batch = valid_batch.to(device)
                valid_label = valid_label.to(device)

                valid_predictions.append(model(valid_batch))
                valid_targets.append(valid_label)

        valid_predictions = torch.cat(valid_predictions)
        valid_targets = torch.cat(valid_targets)
        valid_accuracy = accuracy(valid_predictions, valid_targets)
        valid_logger.add_scalar('valid_accuracy', valid_accuracy, global_step=counter)

        # Evaluate the model
        if max_acc_valid < valid_accuracy:
            max_acc_valid = valid_accuracy
            save_model(model)
        print(f'train accuracy: {train_accuracy}, valid accuracy : {valid_accuracy}, num_epoch : {epoch+1}')

    print('\n\n')

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-log', '--log_dir')
    parser.add_argument('-l', '--lr')
    parser.add_argument('-o', '--mom')
    parser.add_argument('-w', '--wd')
    parser.add_argument('-e', '--epoch')
    parser.add_argument('-b', '--batch')
    parser.add_argument('-d', '--data_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

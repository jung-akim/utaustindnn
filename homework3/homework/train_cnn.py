from .models import CNNClassifier, save_model, load_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
from torchvision import transforms
import torch.utils.tensorboard as tb
import multiprocessing
import numpy as np

def train(model, train_data, valid_data, device, n_epochs, optimizer, logdir, scheduler, retrain = None):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(logdir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(logdir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    """
    if retrain is None:
        model = model.to(device)
    else:
        model = load_model('cnn').to(device)

    loss = torch.nn.CrossEntropyLoss()
    accuracy = lambda output, label: (torch.argmax(output, dim = 1).long() == label.long()).float().mean()
    max_acc_train, max_acc_valid = 0, 0

    # Set up counters and patience
    counter, saved_epoch, patience_counter = 0, 0, 0
    patience = 10

    # Data Augmentation
    crop_size = 32
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          # transforms.RandomCrop(crop_size),
                                          # transforms.Resize(64),
                                          transforms.ColorJitter(brightness=.5, hue=.3)
                                          ])
    # valid_transform = transforms.Compose([transforms.Resize(crop_size+16),
    #                                       transforms.CenterCrop(crop_size),
    #                                       transforms.Resize(64)
    #                                       ])

    for epoch in range(int(n_epochs)):

        # Iterate
        predictions, targets = [], []
        valid_predictions, valid_targets = [], []
        accuracies = []
        model.train()

        for train_batch, train_label in train_data:

            # optimizer.zero_grad()
            # Compute the loss
            train_batch = train_batch.to(device)
            train_label = train_label.to(device)

            # Data Augmentation
            train_batch = train_transform(train_batch)

            o = model(train_batch)
            loss_val = loss.forward(o, train_label)
            accuracies.append(accuracy(o, train_label).detach().cpu().numpy())
            # if counter % 5 == 0:
            #     print(f'loss {loss_val}')

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step() # Gradient becomes NaN if learning rate is too large

            counter += 1
            train_logger.add_scalar('loss', loss_val, global_step=counter)

        train_accuracy = np.mean(accuracies)
        train_logger.add_scalar('train_accuracy', train_accuracy, global_step=counter)

        model.eval()
        val_accuracies = []
        for valid_batch, valid_label in valid_data:
            with torch.no_grad():
                valid_batch = valid_batch.to(device)
                valid_label = valid_label.to(device)

                # Data Augmentation for validation set
                # valid_batch = valid_transform(valid_batch)

                o = model(valid_batch)

                val_accuracies.append(accuracy(o, valid_label).detach().cpu().numpy())

        valid_accuracy = np.mean(val_accuracies)
        valid_logger.add_scalar('valid_accuracy', valid_accuracy, global_step=counter)

        # Log and Update the Learning Rate with scheduler
        train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=counter)
        scheduler.step(valid_accuracy)

        print(f'train accuracy: {train_accuracy}, valid accuracy: {valid_accuracy}, num_epoch: {epoch + 1}')
        # Evaluate the model
        if max_acc_valid < valid_accuracy:
            patience_counter = 0
            max_acc_train = train_accuracy
            max_acc_valid = valid_accuracy
            saved_epoch = epoch + 1
            save_model(model)
            print('Saved the model.')

        patience_counter += 1
        if patience_counter > patience:
            break

    print(f'\nModel at {saved_epoch}th epoch was saved with train accuracy = {max_acc_train} and validation accuracy = {max_acc_valid}')



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
    parser.add_argument('-r', '--retrain')
    # Put custom arguments here

    args = parser.parse_args()

    # Create the CUDA device if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create the ConvNet
    net = CNNClassifier()

    cpu_count = multiprocessing.cpu_count()
    train_data = load_data(args.data_dir + '/train', shuffle=True, batch_size=int(args.batch), num_workers=cpu_count)
    valid_data = load_data(args.data_dir + '/valid', shuffle=False, batch_size=int(args.batch), num_workers=cpu_count)

    optimizer = torch.optim.SGD(net.parameters(),
                                lr = float(args.lr),
                                momentum = float(args.mom),
                                weight_decay = float(args.wd))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 50)

    # Train
    train(net, train_data=train_data, valid_data=valid_data, device=device,
          n_epochs=args.epoch, optimizer=optimizer, logdir=args.log_dir, scheduler = scheduler, retrain=args.retrain)

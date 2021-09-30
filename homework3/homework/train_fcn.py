import torch
import numpy as np

from .models import FCN, save_model, load_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
import multiprocessing


def train(model, train_data, valid_data, device, n_epochs, optimizer, logdir, retrain = None):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(logdir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(logdir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    if retrain is None:
        model = model.to(device)
    else:
        model = load_model('fcn').to(device)

    loss = torch.nn.CrossEntropyLoss(weight = (1 / torch.FloatTensor(DENSE_CLASS_DISTRIBUTION)).to(device))
    max_acc_train, max_acc_valid = 0, 0

    # Set up counters and patience
    counter, saved_epoch, patience_counter = 0, 0, 0
    patience = 1000

    for epoch in range(int(n_epochs)):
        conf = ConfusionMatrix()
        # Iterate
        IOUs = []
        model.train()

        print("Loss: ", end = ' ')
        for train_batch, train_label in train_data:

            # Compute the loss
            train_batch = train_batch.to(device)
            train_label = train_label.to(device)

            o = model(train_batch)
            loss_val = loss.forward(o, train_label.long())
            conf.add(o.argmax(1), train_label)

            IOUs.append(conf.iou.detach().cpu().numpy())
            if counter % 10 == 0:
                print(f'{loss_val:.6f}', end = ' -> ')

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step() # Gradient becomes NaN if learning rate is too large

            counter += 1
            train_logger.add_scalar('loss', loss_val, global_step=counter)
            train_logger.add_histogram('net8', model.net8.weight.grad, global_step=counter)
            train_logger.add_histogram('down3', model.down3.weight.grad, global_step=counter)
            train_logger.add_histogram('net7', model.net7.weight.grad, global_step=counter)
            train_logger.add_histogram('down2', model.down2.weight.grad, global_step=counter)
            train_logger.add_histogram('net6', model.net6.weight.grad, global_step=counter)
            train_logger.add_histogram('down1', model.down1.weight.grad, global_step=counter)

            train_logger.add_histogram('net5-conv1', model.net5.net[0].weight.grad, global_step=counter)
            train_logger.add_histogram('net5-conv2', model.net5.net[3].weight.grad, global_step=counter)

            train_logger.add_histogram('net4-conv1', model.net4.net[0].weight.grad, global_step=counter)
            train_logger.add_histogram('net4-conv2', model.net4.net[3].weight.grad, global_step=counter)

            train_logger.add_histogram('net3-conv1', model.net3.net[0].weight.grad, global_step=counter)
            train_logger.add_histogram('net3-conv2', model.net3.net[3].weight.grad, global_step=counter)

            train_logger.add_histogram('net2-conv1', model.net2.net[0].weight.grad, global_step=counter)
            train_logger.add_histogram('net2-conv2', model.net2.net[3].weight.grad, global_step=counter)

            train_logger.add_histogram('net1-conv1', model.net1.net[0].weight.grad, global_step=counter)
            train_logger.add_histogram('net1-conv2', model.net1.net[3].weight.grad, global_step=counter)


        train_iou = np.mean(IOUs)
        train_logger.add_scalar('train_iou', train_iou, global_step=counter)

        model.eval()
        val_IOUs = []
        conf_val = ConfusionMatrix()
        for valid_batch, valid_label in valid_data:

            with torch.no_grad():
                valid_batch = valid_batch.to(device)
                valid_label = valid_label.to(device)

                o = model(valid_batch)

                conf_val.add(o.argmax(1), valid_label)
                val_IOUs.append(conf_val.iou.detach().cpu().numpy())

        valid_iou = np.mean(val_IOUs)
        valid_logger.add_scalar('valid_iou', valid_iou, global_step=counter)

        # Log and Update the Learning Rate with scheduler
        # train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=counter)
        # scheduler.step(valid_iou)

        print(f'train iou: {train_iou:.6f}, valid iou: {valid_iou:.6f}, num_epoch: {epoch + 1}')

        # Evaluate the model
        if max_acc_valid < valid_iou:
            patience_counter = 0
            max_acc_train = train_iou
            max_acc_valid = valid_iou
            saved_epoch = epoch + 1
            save_model(model)
            print('Saved the model.')

        patience_counter += 1
        if patience_counter > patience:
            break

    print(f'\nModel at {saved_epoch}th epoch was saved with train accuracy = {max_acc_train} and validation accuracy = {max_acc_valid}')


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Put custom arguments here
    parser.add_argument('-log', '--log_dir')
    parser.add_argument('-l', '--lr')
    parser.add_argument('-o', '--mom')
    parser.add_argument('-w', '--wd')
    parser.add_argument('-e', '--epoch')
    parser.add_argument('-b', '--batch')
    parser.add_argument('-d', '--data_dir')
    parser.add_argument('-r', '--retrain')

    args = parser.parse_args()

    # Create the CUDA device if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create the ConvNet
    net = FCN()

    cpu_count = multiprocessing.cpu_count()

    train_transform = dense_transforms.Compose([
    #                                           dense_transforms.RandomHorizontalFlip(),
    #                                           dense_transforms.ColorJitter(brightness=.5, hue=.3),
    #                                           dense_transforms.RandomCrop(96),
                                                dense_transforms.ToTensor()
                                                ])

    train_data = load_dense_data(args.data_dir + '/train', batch_size=int(args.batch), shuffle = True, num_workers=0, transform = train_transform) # num_workers = 0 to run debugger
    valid_data = load_dense_data(args.data_dir + '/valid', batch_size=int(args.batch), shuffle = False, num_workers=0, transform = train_transform)

    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = float(args.lr),
                                 # momentum = float(args.mom),
                                 weight_decay = float(args.wd))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 15)

    # Train
    train(net, train_data=train_data, valid_data=valid_data, device=device,
          n_epochs=args.epoch, optimizer=optimizer, logdir=args.log_dir, retrain=args.retrain)


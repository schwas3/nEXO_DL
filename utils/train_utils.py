import numpy as np
import torch
import sparseconvnet as scn
from .data_loaders import SparseData, collatefn
import time

def accuracy(true, pred):
    return sum(true.detach().numpy()==pred.detach().numpy())/len(true.detach().numpy())

def train_one_epoch(epoch_id, net, criterion, optimizer, loader, datafile, nclass = 2):
    """
        Trains the net for all the train data one time
    """
    net.train()
    loss_epoch = 0
    acc_epoch = 0
    start = time.time()
    for batchid, (coord, ener, label, event) in enumerate(loader):
        batch_size = len(event)
        ener, label = ener.cuda(), label.cuda()

        optimizer.zero_grad()

        output = net.forward((coord, ener, batch_size))

        loss = criterion(output, label)
        loss.backward()

        optimizer.step()

        loss_epoch += loss.item()

        softmax = torch.nn.Softmax(dim = 1)
        prediction = torch.argmax(softmax(output), 1)
        acc_epoch += accuracy(label.cpu(), prediction.cpu())
    print(time.time() - start)
    loss_epoch = loss_epoch / len(loader)
    acc_epoch = acc_epoch / len(loader)
    epoch_ = f"Train Epoch: {epoch_id}"
    loss_ = f"\t Loss: {loss_epoch:.6f}, Acc: {acc_epoch:.3f}"
    print(epoch_ + loss_)

    return loss_epoch, acc_epoch


def valid_one_epoch(net, criterion, loader, datafile, nclass = 2):
    """
        Computes loss and accuracy for classification
        for all the validation data
    """
    net.eval()
    loss_epoch = 0
    acc_epoch = 0
    start = time.time()
    loader_labels = np.array([])
    loader_predicts = np.array([])

    with torch.autograd.no_grad():
        for batchid, (coord, ener, label, event) in enumerate(loader):
            batch_size = len(event)
            ener, label = ener.cuda(), label.cuda()

            output = net.forward((coord, ener, batch_size))

            loss = criterion(output, label)

            loss_epoch += loss.item()

            softmax = torch.nn.Softmax(dim = 1)
            prediction = torch.argmax(softmax(output), 1)
            acc_epoch += accuracy(label.cpu(), prediction.cpu())
            loader_labels = np.concatenate((loader_labels, label.cpu().detach().numpy()), axis=None)
            loader_predicts = np.concatenate((loader_predicts, softmax(output).cpu().detach().numpy()[:, 1]), axis=None)
        loss_epoch = loss_epoch / len(loader)
        acc_epoch = acc_epoch / len(loader)
        loss_ = f"\t Validation Loss: {loss_epoch:.6f} Accu: {acc_epoch:.3f}"
        print(loss_, time.time() - start)
    with open('valid.npy', 'wb') as f:
        np.save(f, np.array([loader_labels, loader_predicts]))

    return loss_epoch, acc_epoch

def train_net(*,
              nepoch,
              train_data_path,
              valid_data_path,
              train_batch_size,
              valid_batch_size,
              net,
              criterion,
              optimizer,
              datafile,
              checkpoint_dir,
              num_workers,
              start_loss = np.inf, 
	      nevents_train = None,
              nevents_valid = None):
    """
        Trains the net nepoch times and saves the model anytime the validation loss decreases
    """
    train_gen = SparseData(train_data_path, datafile, nevents = nevents_train)
    valid_gen = SparseData(valid_data_path, datafile, nevents = nevents_valid)

    loader_train = torch.utils.data.DataLoader(train_gen,
                                               batch_size = train_batch_size,
                                               shuffle = False,
                                               num_workers = num_workers,
                                               collate_fn = collatefn,
                                               drop_last = True,
                                               pin_memory = False)
    loader_valid = torch.utils.data.DataLoader(valid_gen,
                                               batch_size = valid_batch_size,
                                               shuffle = False,
                                               num_workers = num_workers,
                                               collate_fn = collatefn,
                                               drop_last = True,
                                               pin_memory = False)

    for i in range(nepoch):
        train_loss, train_met = train_one_epoch(i, net, criterion, optimizer, loader_train, datafile)
        valid_loss, valid_met = valid_one_epoch(net, criterion, loader_valid, datafile)

        if valid_loss < start_loss:
            torch.save({'state_dict': net.state_dict(),
			     'optimizer': optimizer.state_dict(),
			     'loss': valid_loss,
                             'acc': valid_met,
			     'train_loss': train_loss,
                             'train_acc': train_met,
                             'epoch': i,
                             'optimizer': optimizer.state_dict()}, f'{checkpoint_dir}/net_checkpoint_{i}.pth')
            start_loss = valid_loss
        else:
            torch.save({'loss': valid_loss,
                             'acc': valid_met,
			     'train_loss': train_loss,
                             'train_acc': train_met,
                             'epoch': i}, f'{checkpoint_dir}/summary_{i}.pth')
            start_loss = valid_loss



def predict_gen(data_path, net, datafile, batch_size, nevents):
    """
    A generator that yields a dictionary with output of collate plus
    output of  network.
    Parameters:
    ---------
        data_path : str
                    path to dataset
        net       : torch.nn.Model
                    network to use for prediction
        batch_size: int
        nevents   : int
                    Predict on only nevents first events from the dataset
    Yields:
    --------
        dict
            the elements of the dictionary are:
            coords      : np.array (2d) containing XYZ coordinate bin index
            label       : np.array containing original voxel label
            energy      : np.array containing energies per voxel
            dataset_id  : np.array containing dataset_id as in input file
            predictions : np.array (2d) containing predictions for all the classes
    """

    gen    = SparseData(data_path, datafile, nevents = nevents)
    loader = torch.utils.data.DataLoader(gen,
                                         batch_size = batch_size,
                                         shuffle = False,
                                         num_workers = 1,
                                         collate_fn = collatefn,
                                         drop_last = False,
                                         pin_memory = False)

    net.eval()
    softmax = torch.nn.Softmax(dim = 1)
    with torch.autograd.no_grad():
        for batchid, (coord, ener, label, event) in enumerate(loader):
            batch_size = len(event)
            ener, label = ener.cuda(), label.cuda()
            output = net.forward((coord, ener, batch_size))
            y_pred = softmax(output).cpu().detach().numpy()

            out_dict = dict(
                    label = label.cpu().detach().numpy(),
                    dataset_id = event,
                    prediction = y_pred)

            yield out_dict

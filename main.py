import os, sys
import time
import pprint
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchtext
import spacy
import argparse
import matplotlib.pyplot as plt
from rnn import GRU_Layer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spacy = spacy.load('en_core_web_sm')


def parse_args():
    parser = argparse.ArgumentParser(description='hyperparameter setting')

    # general setting
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--epoch_num', dest='epoch_num', type=int, default=10)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)

    # model setting
    parser.add_argument('--model', dest='model', type=str, default='gru')

    # rnn setting
    parser.add_argument('--rnn', dest='rnn', type=str, default='GRU')
    parser.add_argument('--bidirection', dest='bidirection', type=bool, default=False)
    parser.add_argument('--input_size', dest='input_size', type=int, default=300)
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', dest='num_layers', type=int, default=1)
    parser.add_argument('--output_size', dest='output_size', type=int, default=2)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.2)

    # learing rate setting(ReduceLROnPlateau(mode='min'))
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=math.sqrt(1e-5))
    parser.add_argument('--min_learning_rate', dest='min_learning_rate', type=float, default=math.sqrt(1e-20))
    parser.add_argument('--patience', dest='patience', type=int, default=5)
    parser.add_argument('--cooldown', dest='cooldown', type=int, default=5)
    parser.add_argument('--factor', dest='factor', type=float, default=math.sqrt(0.1))

    # optimizer setting
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-3)

    # vocabulary setting
    parser.add_argument('--min_freq', dest='min_freq', type=int, default=10)
    parser.add_argument('--fix_length', dest='fix_length', type=int, default=64)

    args = parser.parse_args()
    return args

def sec2str(sec):
    if sec < 60:
        return "elapsed: {:02d}s".format(int(sec))
    elif sec < 3600:
        min = int(sec / 60)
        sec = int(sec - min * 60)
        return "elapsed: {:02d}m{:02d}s".format(min, sec)
    elif sec < 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        return "elapsed: {:02d}h{:02d}m{:02d}s".format(hr, min, sec)
    elif sec < 365 * 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        dy = int(hr / 24)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        hr = int(hr - dy * 24)
        return "elapsed: {:02d} days, {:02d}h{:02d}m{:02d}s".format(dy, hr, min, sec)

def train(train_iter, val_iter, net, criterion, optimizer, lr_scheduler, TEXT, args):
    print('start train and validation')
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_loss = 100
    result_dir = os.path.join('./result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    checkpoint_dir = os.path.join('./checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    for epoch in range(args.epoch_num):
        start = time.time()
        train_loss = train_acc = val_loss = val_acc = 0
        net.train()
        print('epoch', epoch+1)
        for i, batch_train in enumerate(train_iter):
            text = batch_train.text
            label = batch_train.label
            # print(text.size())
            # print(label.size())
            # print('text')
            # print(text)
            # print('label')
            # print(label)
            if text.size(0) != args.batch_size:
                break

            text = text.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = net(text)
            # print('output')
            # print(output.size())
            # print(output)
            # print(output.max(1))
            # print('label')
            # print(label.size())
            loss = criterion(output, label)
            train_loss += loss.item()
            # print(train_loss)
            train_acc += (output.max(1)[1] == label).sum().item()
            # print(train_acc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()
            # break
        lr_scheduler.step(loss)
        avg_train_loss = train_loss / len(train_iter.dataset)
        avg_train_acc = train_acc / len(train_iter.dataset)
        # print('loss', avg_train_loss)
        print('train', sec2str(time.time() - start))
        # break

        start = time.time()
        net.eval()
        with torch.no_grad():
            for i, batch_val in enumerate(val_iter):
                text = batch_val.text
                label = batch_val.label
                if text.size(0) != args.batch_size:
                    break
                text = text.to(device)
                label = label.to(device)

                output = net(text)
                
                loss = criterion(output, label)
                val_loss += loss.item()
                val_acc += (output.max(1)[1] == label).sum().item()
        avg_val_loss = val_loss / len(val_iter.dataset)
        avg_val_acc = val_acc / len(val_iter.dataset)

        print('validation', sec2str(time.time() - start))
        print('Epoch [{}/{}], train_loss: {loss:.4f}, train_acc: {acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}' 
                    .format(epoch+1, args.epoch_num, loss=avg_train_loss, acc=avg_train_acc, val_loss=avg_val_loss, val_acc=avg_val_acc))
        
        if avg_val_loss <= best_loss:
            print('save parameters')
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'checkpoint.pth'))
            best_loss = avg_val_loss
        
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)
        # break
    
    plt.figure()
    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='val')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'loss.png'))
    plt.figure()
    plt.plot(train_acc_list, label='train')
    plt.plot(val_acc_list, label='val')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'acc.png'))

def test(test_iter, net, TEXT, args):
    print('start test')
    start = time.time()
    save_dir = os.path.join('./result/heatmap/test')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join('./checkpoint', 'checkpoint.pth')
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()
    with torch.no_grad():
        total = 0
        test_acc = 0
        for i, batch_test in enumerate(test_iter):
            text = batch_test.text
            label = batch_test.label
            if text.size(0) != args.batch_size:
                break
            text = text.to(device)
            label = label.to(device)

            output = net(text)
            
            test_acc += (output.max(1)[1] == label).sum().item()
            total += label.size(0)
    print('精度: {} %'.format(100 * test_acc / total))
    print('test', sec2str(time.time() - start))


def main():
    args = parse_args()
    pprint.pprint(args)

    TEXT = torchtext.data.Field(sequential=True, tokenize='spacy', lower=True, fix_length=args.fix_length, batch_first=True, include_lengths=False)
    LABEL = torchtext.data.LabelField()

    start = time.time()
    print('Loading ...')

    train_dataset, test_dataset = torchtext.datasets.IMDB.splits(TEXT, LABEL, root='./data')
    print('train dataset', len(train_dataset))
    print('test dataset', len(test_dataset))
    print('Loading time', sec2str(time.time() - start))
    test_dataset, val_dataset = test_dataset.split()

    TEXT.build_vocab(train_dataset, min_freq=args.min_freq, vectors=torchtext.vocab.GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_dataset)

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits((train_dataset, val_dataset, test_dataset),
                                                                        batch_size=args.batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    print('train_iter {}, val_iter {}, test_iter {}'.format(len(train_iter.dataset), len(val_iter.dataset), len(test_iter.dataset)))
    word_embeddings = TEXT.vocab.vectors
    print('word embbedings', word_embeddings.size())

    print(args.model)
    net = GRU_Layer(word_embeddings, args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, verbose=True, min_lr=args.min_learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.factor)

    train(train_iter, val_iter, net, criterion, optimizer, lr_scheduler, TEXT, args)
    test(test_iter, net, TEXT, args)
    print('finished', sec2str(time.time() - start))


if __name__ == '__main__':
    main()
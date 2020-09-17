import argparse
import logging
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from transformers import BertTokenizer, BertModel

from card_image.dataset import MonsterDataset, data_transform


img_dir = './card_image/im/'
dir_checkpoint = 'vgg_ckpt/'


class classifier(nn.Module):
    def __init__(self, feature, withname=True):
        super(classifier, self).__init__()
        self.feature = feature
        self.withname = withname

        self.conv = nn.Sequential(
            nn.Conv2d(512, 2048, kernel_size=7, padding=0),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        if withname:
            self.name_layer = nn.Sequential(
                nn.Linear(768, 384),
                nn.BatchNorm1d(384),
                nn.ReLU(inplace=True)
            )
            self.race_layer = nn.Linear(2048+384, 25)
            self.attr_layer = nn.Linear(2048+384, 7)
        else:
            self.race_layer = nn.Linear(2048, 25)
            self.attr_layer = nn.Linear(2048, 7)

        nn.init.normal_(self.race_layer.weight.data, 0, 0.01)
        nn.init.normal_(self.attr_layer.weight.data, 0, 0.01)
    
    def forward(self, x, y=None):
        x = self.feature(x)
        x = self.conv(x)
        x = self.GAP(x)
        x = torch.flatten(x, 1)

        if self.withname:
            y = self.name_layer(y)
            x = torch.cat([x,y], dim=1)

        race = self.race_layer(x)
        attr = self.attr_layer(x)
        return race, attr


def train(
    net,
    device,
    epochs=5,
    batch_size=1,
    lr=0.001,
    val_percent=0.1,
    per_val_epoch=1,
    withname=False,
):
    dataset = MonsterDataset(img_dir, transform=data_transform, item_list=['id', 'name', 'race', 'attribute'])
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    if withname:
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        bertmodel = BertModel.from_pretrained("bert-base-chinese")
        bertmodel.to(device=device)

    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        net.train()

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                name = batch['name']
                imgs = batch['image']
                race = batch['race'] - 1
                attribute = batch['attribute'] - 1

                if withname:
                    input_ids = []
                    for name_i in name:
                        input_ids.append(torch.tensor([tokenizer.encode(name_i, add_special_tokens=True, max_length=32, truncation=True, padding='max_length')]))
                    input_ids = torch.cat(input_ids, 0).to(device=device)
                    with torch.no_grad():
                        feat = bertmodel(input_ids)[1]

                imgs = imgs.to(device=device, dtype=torch.float32)
                race = race.to(device=device, dtype=torch.long)
                attribute = attribute.to(device=device, dtype=torch.long)

                prace, pattr = net(imgs, feat)
                loss1 = criterion(prace, race)
                loss2 = criterion(pattr, attribute)
                loss = loss1 + loss2
                race_acc = torch.eq(prace.argmax(dim=1), race).sum().float().item() / len(name)
                attr_acc = torch.eq(pattr.argmax(dim=1), attribute).sum().float().item() / len(name)

                pbar.set_postfix(**{'loss race (batch)': loss1.item(), 'loss attr (batch)': loss2.item(), 'acc race (batch)': race_acc, 'acc attr (batch)': attr_acc})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])

        if (epoch+1) % per_val_epoch == 0:
            loss1_total = 0
            loss2_total = 0
            acc_race = 0
            acc_attr = 0
            race_topk = 0
            attr_topk = 0
            net.eval()
            for batch in val_loader:
                name = batch['name']
                imgs = batch['image']
                race = batch['race'] - 1
                attribute = batch['attribute'] - 1

                if withname:
                    input_ids = []
                    for name_i in name:
                        input_ids.append(torch.tensor([tokenizer.encode(name_i, add_special_tokens=True, max_length=32, truncation=True, padding='max_length')]))
                    input_ids = torch.cat(input_ids, 0).to(device=device)
                    with torch.no_grad():
                        feat = bertmodel(input_ids)[1]

                imgs = imgs.to(device=device, dtype=torch.float32)
                race = race.to(device=device, dtype=torch.long)
                attribute = attribute.to(device=device, dtype=torch.long)

                with torch.no_grad():
                    prace, pattr = net(imgs, feat)
                    loss1_total += criterion(prace, race)
                    loss2_total += criterion(pattr, attribute)
                    acc_race += torch.eq(prace.argmax(dim=1), race).sum().float().item()
                    acc_attr += torch.eq(pattr.argmax(dim=1), attribute).sum().float().item()

                    _, racek = torch.topk(prace, k=3, dim=-1)
                    _, attrk = torch.topk(pattr, k=3, dim=-1)
                    race_topk += (race.view(-1,1) == racek).sum().item()
                    attr_topk += (attribute.view(-1,1) == attrk).sum().item()

            logging.info('Validation\n\tLoss: race={}, attr={}\n\tAccuracy: race={}, attr={}\n\tTop 3 Accuracy: race={}, attr={}'.format(
                loss1_total/n_val, loss2_total/n_val, acc_race/n_val, acc_attr/n_val, race_topk/n_val, attr_topk/n_val))
            
            # view result
            race_topk = torch.topk(prace, k=3).indices.cpu().numpy()+1
            attr_topk = torch.topk(pattr, k=3).indices.cpu().numpy()+1
            ids = batch['id']
            for i, id_item in enumerate(ids):
                v = dataset.show_item(id=str(id_item.item()))
                print('{}\tlabel={}, top3=[{}, {}, {}]\tlabel={}, top3=[{}, {}, {}]'.format(
                    v['name'],
                    v['race'], dataset.static_dict['race'][race_topk[i,0]], dataset.static_dict['race'][race_topk[i,1]], dataset.static_dict['race'][race_topk[i,2]],
                    v['attribute'], dataset.static_dict['attribute'][attr_topk[i,0]], dataset.static_dict['attribute'][attr_topk[i,1]], dataset.static_dict['attribute'][attr_topk[i,2]]))
            
        try:
            os.mkdir(dir_checkpoint)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save(net.state_dict(),
                    dir_checkpoint + f'CP_epoch.pth')
        logging.info(f'Checkpoint {epoch + 1} saved !')


def get_args():
    parser = argparse.ArgumentParser(description='Train the VGG on images and target label',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-n', '--withname', dest='withname', type=bool, default=False,
                        help='Use card name for classification')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    feat = models.vgg19_bn(pretrained=True).features
    
    net = classifier(feat, withname=args.withname)
    net.to(device=device)
    
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        print('[*] Load model from', args.load)

    try:
        train(
            net=net,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            device=device,
            val_percent=args.val/100,
            per_val_epoch=1,
            withname=args.withname,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
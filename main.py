from torch.utils.data import DataLoader
import argparse
import PIL
import io
import os.path as osp
import tensorflow as tf
from dataset import FlowerDataset, train_transforms, val_transforms
from models import PlainNet34, ResNet34, ResNet50, PretrainedResNet50, RepVGG_AO, RepVGG_B3, PretrainedResnet34, Wide_resnet50_2
from train import fit
from tqdm import tqdm
from PIL import Image
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument(
        '--net_type',
        choices=['plain34', 'resnet34', 'resnet50', 'pretrained_resnet50', 'RepVGG_A0', 'RepVGG_B3', 'pretrained_resnet34', 'wide_resnet50_2'],
        help='net type')
    args = parser.parse_args()
    return args

def load_image(csv_file):
    data = pd.read_csv(csv_file)
    img, id, label = data['x'], data['id'], data['label']
    img = img.values.tolist()
    id = id.values.tolist()
    label = label.values.tolist()

    imgs = []
    for i in range(len(img)):
        img_path = osp.join('./tpu-getting-started/Image',img[i])
        # image = cv2.imread(img_path)
        # imgs.append(image)
        imgs.append(img_path)
    return id, label, imgs

def save_images(ids,classes,images,save_path):
    for id,cls,im in tqdm(zip(ids,classes,images)):
        path = osp.join(save_path,id+'_'+str(cls)+'.png')
        img = PIL.Image.open(io.BytesIO(im))
        img.save(path)

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    args = parse_args()

    # load dataset
    print('\nLoad dataset')
    # train_files = glob.glob('./tpu-getting-started/*224/train/*.tfrec')
    # val_files = glob.glob('./tpu-getting-started/*224/val/*.tfrec')

    # train_ids, train_class, train_images = convert_tfrecord(train_files, mode='train')
    # val_ids, val_class, val_images = convert_tfrecord(val_files, mode='val')

    # train_path = './tpu-getting-started/trainImage'
    # save_images(train_ids, train_class, train_images, train_path)
    # val_path = './tpu-getting-started/valImage'
    # save_images(val_ids, val_class, val_images, val_path)

    # log_dir = osp.join('tb_logs', 'resnet50')
    # if osp.isdir(log_dir):
    #     shutil.rmtree(log_dir)
    # writer = SummaryWriter(log_dir=log_dir)

    # load net
    print('\nLoad {}'.format(args.net_type))
    if args.net_type == 'plain34':
        net = PlainNet34()
    elif args.net_type == 'resnet34':
        net = ResNet34()
    elif args.net_type == 'resnet50':
        net = ResNet50()
    elif args.net_type == 'pretrained_resnet50':
        net = PretrainedResNet50()
    elif args.net_type == 'RepVGG_A0':
        net = RepVGG_AO()
    elif args.net_type == 'RepVGG_B3':
        net = RepVGG_B3()
    elif args.net_type == 'pretrained_resnet34':
        net = PretrainedResnet34()
    elif args.net_type == 'wide_resnet50_2':
        net = Wide_resnet50_2()

    epochs = 70
    train_loss_sum, val_loss_sum = 0, 0
    train_acc_sum, val_acc_sum = 0, 0
    # train_acc = []
    # train_losses = []
    # val_acc = []
    # val_losses = []
    for i in range(5):
        train_transform, val_transform = train_transforms(args.net_type), val_transforms(args.net_type)
        csv_path = './tpu-getting-started/csv'
        train_csv = osp.join(csv_path, 'train_'+str(i)+'.csv')
        val_csv = osp.join(csv_path, 'val_' + str(i) + '.csv')
        train_ids, train_class, train_images = load_image(train_csv)
        val_ids, val_class, val_images = load_image(val_csv)
        train_ds = FlowerDataset(train_ids, train_class, train_images, train_transform, mode='train')
        val_ds = FlowerDataset(val_ids, val_class, val_images, val_transform, mode='val')

        ds_size = dict(
            train=len(train_ds),
            val=len(val_ds))

        loaders = dict(
            train=DataLoader(train_ds, 32, num_workers=4, pin_memory=True),
            val=DataLoader(val_ds, 32, num_workers=4, pin_memory=True)
        )

        print(f'train {i}')
        log_path = osp.join('tb_logs', args.net_type)
        log_dir = osp.join(log_path, 'train'+str(i))
        train_accuracy, train_loss, val_accuracy, val_loss = fit(args.net_type, loaders, ds_size, log_dir, net, epochs, device='cuda:0')
        train_loss_sum += train_loss
        train_acc_sum += train_accuracy
        val_loss_sum += val_loss
        val_acc_sum += val_accuracy

    train_loss_average, train_accuracy_average = train_loss_sum / 5, train_acc_sum / 5
    val_loss_average, val_accuracy_average = val_loss_sum / 5, val_acc_sum / 5
    print('train_loss_average:', train_loss_average, 'train_accuracy_average:', train_accuracy_average)
    print('val_loss_average:', val_loss_average, 'val_accuracy_average:', val_accuracy_average)



import argparse
from models import PretrainedResNet50, RepVGG_AO, RepVGG_B3, PretrainedResnet34, Wide_resnet50_2
from torch.utils.data import DataLoader, Dataset
import os
import torch
import pandas as pd
from tqdm import tqdm
import os.path as osp
from dataset import FlowerDataset, test_transforms
import torch.nn.functional as F
import numpy as np

# def parse_args():
#     parser = argparse.ArgumentParser(description='Train classification network')
#     parser.add_argument(
#         '--net_type',
#         choices=['plain34', 'resnet34', 'resnet50', 'pretrained_resnet50', 'RepVGG_AO', 'RepVGG_B3'],
#         help='net type')
#     args = parser.parse_args()
#     return args

# def save_images(ids,images,save_path):
#     for id,im in tqdm(zip(ids,images)):
#         path = osp.join(save_path,id+'.png')
#         img = PIL.Image.open(io.BytesIO(im))
#         img.save(path)

def load_img(img_path):
    imgs = []
    ids = []
    X = os.listdir(img_path)
    for i in range(len(X)):
        id = X[i].split('.')[0]
        img = osp.join(img_path, X[i])
        ids.append(id)
        imgs.append(img)
    return ids, imgs

def test(test_loader, model):
    # results = []
    predict = []
    model.eval()
    for batch, _, ids in tqdm(test_loader):
        with torch.no_grad():
            # if cuda:
            #     batch = batch.cuda()
            model.eval()
            out = model(batch)
            out = F.softmax(out)
            predict.append(out)

    predict = torch.cat(predict, dim=0)
    return predict

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # load model weights
    weights_path = ["./checkpoints/wide_resnet50_2/069.pth",
                    "./checkpoints/pretrained_resnet34/062.pth",
                    "./checkpoints/pretrained_resnet50/069.pth",
                    "./checkpoints/RepVGG_A0/069.pth", "./checkpoints/RepVGG_B3/041.pth"]
    module = [Wide_resnet50_2(), PretrainedResnet34(), PretrainedResNet50(), RepVGG_AO(), RepVGG_B3()]

    # prediction
    # test_files = glob.glob('./tpu-getting-started/*224/test/*.tfrec')
    # test_ids, test_images = convert_tfrecord(test_files, mode='test')
    # test_path = './tpu-getting-started/testImage'
    # save_images(test_ids, test_images, test_path)
    batch_size = 32
    test_transform = test_transforms()
    test_path = './tpu-getting-started/testImage'
    test_ids, test_images = load_img(test_path)
    test_ds = FlowerDataset(test_ids, None, test_images, test_transform, mode='test')
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    num_classes = 104
    output = torch.zeros(len(test_images), num_classes)
    for i in range(5):
        weight_path = weights_path[i]
        assert os.path.exists(weight_path), "file: '{}' dose not exist.".format(weight_path)
        module[i].load_state_dict(torch.load(weight_path, map_location=device)['net'])

        out = test(test_loader, module[i])
        output = torch.add(out, output)

    output = output / 5
    results = []
    for i, ids in enumerate(test_ids):
        pred_labels = torch.argmax(output[i].data.cpu())
        rows =[ids, pred_labels.numpy().tolist()]
        rows = np.array(rows).reshape(1, 2)
        results.append(pd.DataFrame(rows, columns=['id', 'label']))

    result_df = pd.concat(results)
    result_df['label'] = result_df['label'].astype(int)
    result_df.to_csv('./result_kfold.csv', index=False)


if __name__ == '__main__':
    main()

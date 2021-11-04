import io
import random

from PIL import Image
import torchvision

class FlowerDataset():
    """PyTorch Dataset for flower dataset
    """

    def __init__(self, id_list, class_list, image_list, transforms, mode='train'):
        self.id_list = id_list
        self.class_list = class_list if mode != 'test' else None
        self.image_list = image_list
        self.transforms = transforms
        self.mode = mode
    
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, ind):
        img = self.image_list[ind]
        # img = Image.open(io.BytesIO(img))
        # print("========================",img,"==============================")
        img = Image.open(img)
        img = self.transforms(img)
        if self.mode == 'test':
            return img, -1, self.id_list[ind]
        return img, int(self.class_list[ind]), self.id_list[ind]


def train_transforms(model_type):
    """Transform for train, validation dataset

    1. The image is resized  with its shorter side randomly sampled in [256, 480] for scale augmentation
        -> torchvision.transforms.Resize(size) # size : [256, 480]

    2. 224x224 crop is randomly sampled from an image
        -> torchvision.transforms.RandomResizedCrop(224)

    3. Random horizontal flip
        -> torchvision.transforms.RandomHorizontalFlip(p=0.5)

    4. Per-pixel mean subtracted
        - For pretrained resnet50, ImageNet mean, std are required
            -> torchvision.transforms.ToTensor()
            -> torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        - For scratch training for plain34, resnet34, resnet50, flower dataset mean, std are required
            >>> python test/test_mean_std.py
            |channel:1| - mean: 0.45327600836753845, std: 0.27964890003204346
            |channel:2| - mean: 0.4157607853412628, std: 0.2421468198299408
            |channel:3| - mean: 0.3070197105407715, std: 0.2701781690120697

            -> torchvision.transforms.ToTensor()
            -> torchvision.transforms.Normalize(mean=[0.453, 0.416, 0.307], std=[0.280, 0.242, 0.270])

    5. Standard color augmentation is used
        - Not apply color augmentation, I thought color is an important feature for classification of flowers
    """

    assert model_type in ['plain34', 'resnet34', 'resnet50', 'pretrained_resnet50', 'RepVGG_A0', 'RepVGG_B3', 'pretrained_resnet34', 'wide_resnet50_2']

    rand_size = random.randint(256, 480)
    if model_type == 'pretrained_resnet50':
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else: # transform for plain34, resnet34, resnet50
        normalize = torchvision.transforms.Normalize(mean=[0.453, 0.416, 0.307], std=[0.280, 0.242, 0.270])

    return torchvision.transforms.Compose([
               torchvision.transforms.Resize(rand_size),
               torchvision.transforms.RandomResizedCrop(224),
               torchvision.transforms.RandomHorizontalFlip(p=0.5),
               torchvision.transforms.ToTensor(),
               normalize
            ])


def val_transforms(model_type):
    assert model_type in ['plain34', 'resnet34', 'resnet50', 'pretrained_resnet50', 'RepVGG_A0', 'RepVGG_B3', 'pretrained_resnet34', 'wide_resnet50_2']
    if model_type == 'pretrained_resnet50':
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else: # transform for plain34, resnet34, resnet50
        normalize = torchvision.transforms.Normalize(mean=[0.453, 0.416, 0.307], std=[0.280, 0.242, 0.270])

    return torchvision.transforms.Compose([
               torchvision.transforms.CenterCrop(224),
               torchvision.transforms.Resize(224),
               torchvision.transforms.ToTensor(),
               normalize
            ])
        

def test_transforms():
    # assert model_type in ['plain34', 'resnet34', 'resnet50', 'pretrained_resnet50', 'RepVGG']
    # if model_type == 'pretrained_resnet50':
    #     normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # else:  # transform for plain34, resnet34, resnet50
    normalize = torchvision.transforms.Normalize(mean=[0.453, 0.416, 0.307], std=[0.280, 0.242, 0.270])
    
    return torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize
    ])

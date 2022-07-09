import argparse
import os
import torch
import tqdm
import numpy as np
from PIL import Image

from torchvision import transforms

import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    def __init__(self, resnet_version='18', trainable=True):
        """Load the pretrained ResNet-152 and replace top fc layer.
        """
        super(EncoderCNN, self).__init__()
        if resnet_version == '18':
            self.cnn = models.resnet18(pretrained=True)
            self.feature_dim = 512
        elif resnet_version == '34':
            self.cnn = models.resnet34(pretrained=True)
            self.feature_dim = 512
        elif resnet_version == '50':
            self.cnn = models.resnet50(pretrained=True)
            self.feature_dim = 2048   
        else:
            raise NotImplementedError

        del self.cnn.avgpool
        del self.cnn.fc

        self.ln = nn.LayerNorm(self.feature_dim)

        if not trainable:
            self.cnn.eval()
            for param in self.cnn.parameters():
                param.requires_grad = False

    def forward(self, images):
        """Extract the image feature vectors."""
        # See note [TorchScript super()]
        x = self.cnn.conv1(images)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)

        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, self.feature_dim)

        x = self.ln(x)
        return x


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, image_paths, transform=lambda x: x):
        self.image_ids = image_ids
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path).convert('RGB') as img:
            return self.transform(img), self.image_ids[idx]


def main(args):
    
    with open(os.path.join(args.processed_dir, f'{args.split}.image'), 'r') as f:
        image_names = f.readlines()
        image_names = set([img_name.strip() for img_name in image_names])

    image_names = list(image_names)
    image_paths = [os.path.join(args.image_dir, image_name) for image_name in image_names]
    features_dir = os.path.join(args.processed_dir, f'{args.split}-features-grid')

    os.makedirs(features_dir, exist_ok=True)

    resnet = EncoderCNN('50', False)
    resnet.eval()
    resnet.to(args.device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(image_names, image_paths, transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         pin_memory=args.device.type == 'cuda',
                                         shuffle=False)

    with torch.no_grad():
        for imgs, ids in tqdm.tqdm(loader):
            outs = resnet(imgs.to(args.device))
            for out, id in zip(outs, ids):
                out = out.cpu().numpy()

                # id = str(id.item())
                np.save(os.path.join(features_dir, id), out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-COCO captions pre-processing.')

    parser.add_argument('--image-dir', type=str)
    parser.add_argument('--split', choices=['train', 'valid', 'test'], help="Data split ('train', 'valid' or 'test').")
    parser.add_argument('--processed-dir', type=str)
    parser.add_argument('--device', default='cuda', type=torch.device, help="Device to use ('cpu', 'cuda', ...).")
    parser.add_argument('--batch-size', default=8, type=int, help="Image batch size.")
    parser.add_argument('--num-workers', default=0, type=int, help="Number of data loader workers.")

    main(parser.parse_args())

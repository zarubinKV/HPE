from PIL import Image
import numpy as np
import cv2
import timm

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn

from yolo import YOLO


config = {
    'weight': 'data/model_final.pt',
    'finger': 8
}

# Change your mean and std
mean = [0.50588235, 0.50196078, 0.22941176]
std = [0.21176471, 0.21568627, 0.20392157]

image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

art_line = []


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def show_predict(dataset):
    """
    Function to visualize data
    Input: torch.utils.data.Dataset
    """
    unorm = UnNormalize(mean=mean, std=std)

    image = dataset.__getitem__(1)
    image = image[:, 0:128, 0:128]
    pred = model(torch.tensor(image).unsqueeze(0).to(device))
    pred = pred.cpu().detach().numpy()[0]

    keypoints = pred.reshape([21, 2])
    keypoints = keypoints * 480

    return keypoints


class FreiHAND(Dataset):
    """Class to load FreiHAND dataset."""

    def __init__(self, image):
        self.image = image
        self.len = 1

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean,
                    std=std,
                ),
            ]
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_raw = self.image
        image = self.image_transform(image_raw)
        image = np.asarray(image)

        return image


def show_wcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:, 0:480, 0:480]

        # Hand detect
        width, height, inference_time, results = yolo.inference(frame)
        results.sort(key=lambda x: x[2])
        x1, y1 = 0, 0
        x2, y2 = 480, 480
        cx, cy = 240, 240
        image_for_cut = frame

        if len(results) > 0:
            detection = results[0]
            _, _, _, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)

            image_for_cut = np.concatenate((np.concatenate((frame, frame), axis=0), frame), axis=0)
            image_for_cut = np.concatenate((np.concatenate((image_for_cut, image_for_cut), axis=1), image_for_cut), axis=1)
            x1 = int(cy + 240)
            y1 = int(cx + 240)
            x2 = int(x1 + 480)
            y2 = int(y1 + 480)

            image = cv2.cvtColor(image_for_cut[x1:x2, y1:y2], cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            img_data = FreiHAND(image)

            keypoints = show_predict(img_data)
            y, x = keypoints[:, 0].astype(int), keypoints[:, 1].astype(int)

            x = int(x[config['finger']] - 240 + cy)
            y = int(y[config['finger']] - 240 + cx)

            art_line.append([y, x])
            frame[x - 2: x + 2, y - 2: y + 2] = [0, 0, 255]

        if len(art_line) > 1:
            for i in range(1, len(art_line)):
                cv2.line(frame, art_line[i - 1], art_line[i], (255, 0, 0), thickness=2)


        cv2.imshow('frame', frame[:,::-1])

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    device = 'cpu'

    model = timm.create_model('mobilenetv2_050', pretrained=True, num_classes=0)
    model.global_pool = nn.Flatten()
    model.classifier = nn.Sequential(
        nn.Linear(20480, 42),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(config['weight']))
    model.to(device)
    model.eval()

    yolo = YOLO("data/cross-hands-tiny.cfg", "data/cross-hands-tiny.weights", ["hand"])
    yolo.size = int(512)
    yolo.confidence = float(0.2)

    show_wcam()


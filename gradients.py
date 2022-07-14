import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


def squared_difference(prediction, ground_truth):
    difference = torch.abs(prediction - ground_truth) ** 2
    return difference


class Gradient_Analysis(nn.Module):
    def __init__(self, model: nn.Module, layer_list: list, height: int, width: int, reduction: str):
        super().__init__()
        self.model = model
        self.grad_dict = {}
        self.height = height
        self.width = width
        self.stop_time = 0
        self.reduction = reduction

        for layer in layer_list:
            self.grad_dict[layer] = []

        for name, layer in self.model.named_modules():
            layer.__name__ = name
            if name in layer_list:
                layer.register_backward_hook(self.get_feature_map_gradient())

    def get_feature_map_gradient(self):
        def bn(layer, _, grad_output):
            gradients = grad_output[0].cpu().numpy()
            self.stop_time = time.time()
            if gradients.shape[1] == 1:
                gradients = gradients.squeeze()
            else:
                if self.reduction == 'sum':
                    gradients = np.sum(gradients, axis=1)
                elif self.reduction == 'mean':
                    gradients = np.mean(gradients, axis=1)
                elif self.reduction == 'max':
                    gradients = np.max(gradients, axis=1)
                elif self.reduction == 'norm':
                    gradients = np.linalg.norm(gradients, axis=1)
            gradients = (gradients - np.min(gradients)) / (np.max(gradients) - np.min(gradients))
            if gradients.shape[1] != self.height or gradients.shape[2] != self.width:
                resized_grads = []
                for i in range(gradients.shape[0]):
                    temp = cv2.resize(gradients[i], dsize=(self.width, self.height))
                    temp = np.expand_dims(temp, axis=0)
                    resized_grads.append(temp)
                gradients = np.concatenate(resized_grads)
            self.grad_dict[layer.__name__].append(gradients)
        return bn

    def get_gradients(self):
        grad_list = []
        for key in self.grad_dict.keys():
            grad_list.append(np.concatenate(self.grad_dict[key]))
        grad_array = np.array(grad_list)
        if len(grad_list) == 1:
            return grad_array.squeeze(0)
        else:
            mean_grad = np.sum(grad_array, axis=0) / len(grad_array)
            layer_vog = np.sqrt(sum([(mm-mean_grad)**2 for mm in grad_array]) / len(grad_array))
            return layer_vog

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

"""
Implementation of multiple loss functions when multiple segmentation annotations exist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ExtendedDiceLoss(nn.Module):

    def __init__(self):
        super(ExtendedDiceLoss, self).__init__()

    def forward(self, y_pred, y_labels):
        """
        :param y_pred, prediction, [B,C,X,Y], after softmax, before argmax
        :param y_labels, multiple segmentation labels, [B,N,X,Y]
        """
        y_labels = y_labels.long()
        union = torch.zeros_like(y_labels[:,0]).to(y_labels.device)
        intersection = torch.ones_like(y_labels[:,0]).to(y_labels.device)
        for i in range(y_labels.shape[1]):
            union += y_labels[:,i]
            intersection *= y_labels[:,i]
        # get the outer boundary and inner boundary binary label
        y_label_ob = (union > 0).float()
        y_label_ib = (intersection > 0).float()
        error_out = y_pred[:,-1] - y_pred[:,-1] * y_label_ob
        error_in = y_label_ib - y_label_ib * y_pred[:,-1]
        loss = (error_out + error_in).sum() / (y_pred[:,-1] + y_label_ib).sum()
        return loss

# One could also try this version
class ExtendedDiceLoss_v2(nn.Module):

    def __init__(self):
        super(ExtendedDiceLoss_v2, self).__init__()

    def forward(self, y_pred, y_labels):
        """
        :param y_pred, prediction, [B,C,X,Y], after softmax, before argmax
        :param y_labels, multiple segmentation labels, [B,N,X,Y]
        """
        y_labels = y_labels.long()
        union = torch.zeros_like(y_labels[:,0]).to(y_labels.device)
        intersection = torch.ones_like(y_labels[:,0]).to(y_labels.device)
        for i in range(y_labels.shape[1]):
            union += y_labels[:,i]
            intersection *= y_labels[:,i]
        # get the outer boundary and inner boundary binary label
        y_label_ob = (union > 0).float()
        y_label_ib = (intersection > 0).float()
        error_out = y_pred[:,-1] - y_pred[:,-1] * y_label_ob
        error_in = y_label_ib - y_label_ib * y_pred[:,-1]
        loss = (error_out**2 + error_in**2).sum() / (y_pred[:,-1]**2 + y_label_ib**2).sum()
        return loss

class AverageCELoss(nn.Module):

    def __init__(self):
        super(AverageCELoss, self).__init__()

    def forward(self, y_pred, y_labels):
        """
        :param y_pred, prediction, [B,C,X,Y], before softmax
        :param y_labels, multiple segmentation labels, [B,N,X,Y]
        """
        y_labels = y_labels.long()
        criteria = nn.CrossEntropyLoss()
        loss = []
        for i in range(y_labels.shape[1]):
            loss.append(criteria(y_pred, y_labels[:,i]))
        return torch.mean(torch.stack(loss))

class ConsensusCELoss(nn.Module):

    def __init__(self):
        super(ConsensusCELoss, self).__init__()

    def forward(self, y_pred, y_labels):
        """
        :param y_pred, prediction, [B,C,X,Y], before softmax
        :param y_labels, multiple segmentation labels, [B,N,X,Y]
        """
        y_labels = y_labels.long()
        criteria = nn.CrossEntropyLoss()
        union = torch.zeros_like(y_labels[:,0]).to(y_labels.device)
        for i in range(y_labels.shape[1]):
            union += y_labels[:,i]
        y_consensus = (union > int(y_labels.shape[1]/2)).long()
        loss = criteria(y_pred, y_consensus)
        return loss

def cross_entropy_over_annotators(labels, logits, confusion_matrices):
    losses_all_annotators = []
    predictions = []
    for idx, labels_annotator in enumerate(torch.unbind(labels, axis=1)):
        loss, preds_clipped = sparse_confusion_matrix_softmax_cross_entropy(
            labels=labels_annotator,
            logits = logits,
            confusion_matrix = confusion_matrices[idx,:,:],
        )
        losses_all_annotators.append(loss)
        predictions.append(preds_clipped)
    losses_all_annotators = torch.stack(losses_all_annotators, axis=0)
    predictions = torch.stack(predictions, dim=0)
    consistency_loss = torch.sum(torch.softmax(logits, dim=1) - torch.mean(predictions,dim=0),dim=1).mean()
    return torch.mean(losses_all_annotators), consistency_loss

def sparse_confusion_matrix_softmax_cross_entropy(labels, logits, confusion_matrix):
    preds_true = torch.softmax(logits, dim=1)
    preds_annotator=torch.einsum('bijk,li->bljk',preds_true,confusion_matrix)
    # change the label to one-hot
    shape_labels = labels.shape
    labels = labels.view((shape_labels[0], 1, *shape_labels[1:]))
    labels_onehot = torch.zeros(logits.shape).to(labels.device).scatter_(1,labels,1)
    weights = 1.0 / torch.clamp(torch.sum(torch.matmul(preds_true, labels_onehot),dim=1), 1e-10, 0.99999)
    preds_clipped = torch.clamp(preds_annotator, 1e-10, 0.99999)
    cross_entropy = (torch.sum(-labels_onehot * torch.log(preds_clipped), dim=1)*weights).mean()
    return cross_entropy, preds_clipped

# Modify from paper: Learning from noisy labels by regularized estimation of annotator confusion
class ConfusionMatrixLoss(nn.Module):

    def __init__(self, num_annotators, num_classes, theta=0.01):
        super(ConfusionMatrixLoss, self).__init__()
        w_init = torch.tensor(
            np.stack([6.0 * np.eye(num_classes) - 5.0 for j in range(num_annotators)]),
            dtype=torch.float32
        )
        rho = F.softplus(w_init)
        # make sure fit the confusion matrices into the optimzer before training
        # from itertools import chain
        # optimizer = optim.Adam(chain(model.parameters(),loss.parameters()), lr=args.lr)
        self.confusion_matrices = nn.Parameter(torch.div(rho, torch.sum(rho, dim=-1, keepdim=True)), requires_grad=True)
        self.theta = theta

    def forward(self, y_pred, y_labels):
        """
        :param y_pred, prediction, [B,C,X,Y], before softmax
        :param y_labels, multiple segmentation labels, [B,N,X,Y], should be long dtype
        """
        trace_norm = torch.mean(torch.stack([torch.trace(self.confusion_matrices[i]) for i in range(self.confusion_matrices.shape[0])]))
        ce_loss, _ = cross_entropy_over_annotators(y_labels, y_pred, self.confusion_matrices) 
        loss = ce_loss + self.theta * trace_norm
        return loss

# Refer to paper: Let's agree to disagree: Learning highly debatable multirater labelling
# It's based on confusion matrix so I think the confusion matrix loss should be used here, too.
class ConsistencyLoss(nn.Module):

    def __init__(self, num_annotators, num_classes, theta=0.01):
        super(ConsistencyLoss, self).__init__()
        w_init = torch.tensor(
            np.stack([6.0 * np.eye(num_classes) - 5.0 for j in range(num_annotators)]),
            dtype=torch.float32
        )
        rho = F.softplus(w_init)
        self.confusion_matrices = nn.Parameter(torch.div(rho, torch.sum(rho, dim=-1, keepdim=True)), requires_grad=True)
        self.theta = theta

    def forward(self, y_pred, y_labels):
        """
        :param y_pred, prediction, [B,C,X,Y], before softmax
        :param y_labels, multiple segmentation labels, [B,N,X,Y], should be long dtype
        """
        # get the consensus label
        union = torch.zeros_like(y_labels[:,0]).to(y_labels.device)
        for i in range(y_labels.shape[1]):
            union += y_labels[:,i]
        y_consensus = (union > int(y_labels.shape[1]/2)).long()
        ce_loss = nn.CrossEntropyLoss()
        loss_consensus = ce_loss(y_pred, y_consensus)
        trace_norm = torch.mean(torch.stack([torch.trace(self.confusion_matrices[i]) for i in range(self.confusion_matrices.shape[0])]))
        loss_ce, loss_consistency = cross_entropy_over_annotators(y_labels, y_pred, self.confusion_matrices) 
        loss = loss_consensus + loss_ce + self.theta * trace_norm + loss_consistency
        return loss

# staple: it is a technique that can infer the hidden unique ground truth from multiple annotations
# refer https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1STAPLEImageFilter.html
import SimpleITK as sitk
def get_staple_label():
    staple_filter = sitk.STAPLEImageFilter()
    foregroundValue = 255.0
    threshold = 0.95
    # read multiple annotations using sitk, save them in the labels list
    labels = []
    staple_filter.SetForegroundValue(foregroundValue)
    reference_segmentation_STAPLE_probabilities = staple_filter.Execute(labels)
    reference_segmentation_STAPLE = reference_segmentation_STAPLE_probabilities > threshold
    save_path = ''
    sitk.WriteImage(reference_segmentation_STAPLE*255, save_path)

if __name__ == '__main__':
    import numpy as np
    import PIL.Image
    import os
    label_path1 = 'i:/myocardium_dataset_v2/annotations/labels/label_1/A4C/subject_000'
    label_path2 = 'i:/myocardium_dataset_v2/annotations/labels/label_2/A4C/subject_000'
    label_path3 = 'i:/myocardium_dataset_v2/annotations/labels/label_3/A4C/subject_000'
    label_path4 = 'i:/myocardium_dataset_v2/annotations/labels/label_4/A4C/subject_000'
    label_path5 = 'i:/myocardium_dataset_v2/annotations/labels/label_5/A4C/subject_000'
    files = os.listdir(label_path1)
    label1 = torch.from_numpy(np.asarray(PIL.Image.open(os.path.join(label_path1,files[0]))) / 255)
    label2 = torch.from_numpy(np.asarray(PIL.Image.open(os.path.join(label_path2,files[0]))) / 255)
    label3 = torch.from_numpy(np.asarray(PIL.Image.open(os.path.join(label_path3,files[0]))) / 255)
    label4 = torch.from_numpy(np.asarray(PIL.Image.open(os.path.join(label_path4,files[0]))) / 255)
    label5 = torch.from_numpy(np.asarray(PIL.Image.open(os.path.join(label_path5,files[0]))) / 255)
    labels = torch.stack([label1,label2,label3,label4,label5])[None]
    prediction = torch.zeros([512,512]).float()
    prediction = torch.stack([1-prediction, prediction])[None]
    extend_dice_loss = ExtendedDiceLoss()
    loss = extend_dice_loss(prediction, labels)
    print(f'loss:{loss}')
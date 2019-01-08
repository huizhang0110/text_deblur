import torch
import torch.nn as nn


class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output_imgs, target_imgs):
        loss = self.mse_loss(output_imgs, target_imgs)
        return loss


class L1_Loss(nn.Module):

    def __init__(self):
        super(L1_Loss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, output_imgs, target_imgs):
        loss = self.l1_loss(output_imgs, target_imgs)
        return loss


class MSE_OHEM_Loss(nn.Module):

    def __init__(self):
        super(MSE_OHEM_Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, output_imgs, target_imgs):
        """
        output_imgs: [batch_size, 3/1, H, W]
        target_imgs: [batch_size, 3/1, H, W]
        """
        loss_every_sample = []
        batch_size = output_imgs.size(0)
        for i in range(batch_size):
            output_img = output_imgs[i].view(1, -1)
            target_img = target_imgs[i].view(1, -1)
            positive_mask = (target_img < 0.8).float()
            sample_loss = self.mse_loss(output_img, target_img)
            positive_loss = torch.masked_select(sample_loss, positive_mask.byte())
            negative_loss = torch.masked_select(sample_loss, 1 - positive_mask.byte())

            num_positive = int(positive_mask.sum().data.cpu().item())
            k = num_positive * 3
            num_all = output_img.shape[1]
            if k + num_positive > num_all:
                k = int(num_all - num_positive)
            if k < 10:
                avg_sample_loss = sample_loss.mean()
            else:
                negative_loss_topk, _ = torch.topk(negative_loss, k)
                avg_sample_loss = positive_loss.mean() * 3 + negative_loss_topk.mean()
            loss_every_sample.append(avg_sample_loss)

        return torch.stack(loss_every_sample, 0).mean() 



if __name__ == "__main__":
    x = torch.FloatTensor([[1, 2], [3, 4]]).view(1, 1, 2, 2)
    y = torch.FloatTensor([[1.1, 2.1], [3, 4.1]]).view(1, 1, 2, 2)
    # loss_fn = MSELoss(size_average=False)
    loss_fn = MSE_OHEM_Loss()
    loss = loss_fn(y, x)
    print(loss)


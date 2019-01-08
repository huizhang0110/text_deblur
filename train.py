import torch
import argparse
from model import DeblurModel, load_state, save_state
from input_data import BlurDataset
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data
import loss
import os
import torchvision


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--train_data", default="data_train_53394.txt", type=str, help="training data")
parser.add_argument("--eval_data", default="data_eval_30.txt", type=str, help="evaluate data")
parser.add_argument("--exp_dir", default="experiments/demo", type=str, help="ckpt and log path")
parser.add_argument("--backbone", default="resnet18", type=str)
parser.add_argument("--num_workers", default=16, type=int)
parser.add_argument("--max_epoches", default=100, type=int)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--save_state_every", default=10000, type=int)
parser.add_argument("--optim", default="adam", type=str)
args = parser.parse_args()


if not os.path.exists(args.exp_dir):
    os.makedirs(args.exp_dir)
ckpt_dir = os.path.join(args.exp_dir, "ckpt")

def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] *= 0.1

def evaluate_network(epoch, network, eval_dataloader):
    log_dir = os.path.join(args.exp_dir, str(epoch))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    network.eval()
    cnt = 0
    for orig_images, blur_images in eval_dataloader:
        output_images = network(blur_images.cuda()).cpu()
        for blur_img, output_img, orig_img in zip(blur_images, output_images, orig_images):
            image_tensor = torch.stack((blur_img, output_img, orig_img), 0)
            torchvision.utils.save_image(image_tensor, os.path.join(log_dir, "%d.jpg" % cnt),
                    nrow=3, normalize=True, range=(0, 1), scale_each=True, pad_value=0)
            cnt += 1
    print("=> evaluate finished, saving image to %s" % log_dir)
    network.train()

# === Data input pipeline ===
blur_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()])
orig_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()])

train_dataset = BlurDataset(args.train_data, orig_transform, blur_transform) 
num_trains = len(train_dataset)
print("=> training data: %d, %d step/epoch" % (num_trains, num_trains / args.batch_size))
eval_dataset = BlurDataset(args.eval_data, orig_transform, blur_transform)

train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)
eval_dataloader = data.DataLoader(dataset=eval_dataset, batch_size=10,
        shuffle=False, num_workers=1)

# === Building network and optimizer ===
network = DeblurModel(args.backbone)
network = torch.nn.DataParallel(network).cuda()
network.train()
# loss_fn = loss.MSELoss()
loss_fn = loss.MSE_OHEM_Loss()
loss_fn = torch.nn.DataParallel(loss_fn).cuda()

if args.optim == "adam":
    optimizer = optim.Adam(network.parameters(), lr=3e-4, weight_decay=1e-5)
elif args.optim == "adadelta":
    optimizer = optim.Adadelta(network.parameters(), lr=1.0)
else:
    raise ValueError("Not support such optimizer")

# === training process ===
step = 1
total_loss = 0
print("=> start training")
for epoch in range(args.max_epoches):
    for i, (orig_images, blur_images) in enumerate(train_dataloader):
        orig_images, blur_images = orig_images.cuda(), blur_images.cuda()

        optimizer.zero_grad()
        output_images = network(blur_images)
        loss = loss_fn(output_images, orig_images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % args.print_every == 0:
            avg_loss = total_loss / args.print_every
            total_loss = 0
            lr = optimizer.param_groups[0]["lr"]
            print("epoch: %03d, step: %05d, lr: %.6f, loss: %.5f" % (epoch, step, lr, avg_loss))
        step += 1

    evaluate_network(epoch, network, eval_dataloader)

    save_state(ckpt_dir, epoch, network, optimizer)    

    if epoch in [40, 80]:
        adjust_learning_rate(optimizer)


import glob
import json
import os.path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord
from monai.networks.nets import DenseNet121,EfficientNet,resnet50
# loss = torch.nn.MSELoss
# from monai.metrics import MeanSquaredError
from torch.nn import MSELoss
import tqdm
import torchvision.transforms as transforms
import torch
from scipy.ndimage import zoom
import numpy as np
import wandb

def resize_image(image, new_size):
    """
    Resize a 3D image to new size using scipy's zoom function.
    """
    # 计算缩放因子
    zoom_factor = np.array(new_size) / np.array(image.shape)
    # 使用scipy的zoom函数进行缩放
    return zoom(image, zoom_factor, order=1)


def one_hot_encode_gender_smoking(gender, smoking_status,age):
    # gender: "MALE", "FEMALE"
    # smoking_status: "Unknown", "Ex-smoker", "Smoker"

    gender_encoded = 1       if gender == "MALE" else 0
    smoking_encoded = [0, 0, 0,0]
    smoking_dict = {"Unknown": 0, "Ex-smoker": 1, "Smoker": 2,'Non-smoker':3}
    smoking_encoded[smoking_dict[smoking_status]] = 1

    return [gender_encoded]+[age]  + smoking_encoded
class CTImageDataset(Dataset):
    def __init__(self, image_files, json_filesgt,json_files_other, image_transforms=None):
        self.image_files = image_files
        self.json_filesgt = json_filesgt
        self.json_files_other = json_files_other

        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        json_files_other = self.json_files_other[idx]
        json_filesgt = self.json_filesgt[idx]

        # 加载图像
        image = nib.load(image_file).get_fdata()
        new_depth=384
        depth = image.shape[2]
        start = (depth - new_depth) // 2
        end = start + new_depth

        # 剪裁图像
        image = image[start:end, start:end, start:end]
        image = resize_image(image, (128, 128, 64))
        image = np.expand_dims(image, axis=0)  # 增加一个通道维度

        image = np.array(image, dtype=np.float32)
        image = torch.tensor(image)
        image = self.image_transforms(image)

        json_data = {}
        # 加载json信息
        with open(json_filesgt, 'r') as file:
            data = json.load(file)
            json_data.update(data)
        with open(json_files_other, 'r') as file:
            data = json.load(file)
            json_data.update(data)
        # 提取需要的信息
        survival_time = json_data['survival_time_months']
        event = json_data['event']
        gender = json_data['gender']
        age = json_data['age']
        smoking_status = json_data['smoking_status']

        # 根据需要调整这里的特征
        # features = np.array([event, gender, age, smoking_status], dtype=np.float32)
        # gender = one_hot_encode_gender_smoking(gender,smoking_status)
        gender_and_smoking = one_hot_encode_gender_smoking(gender,smoking_status,age)
        features = np.array(gender_and_smoking ,dtype=np.float32)

        sample = {'image': image, 'features': features, 'label': survival_time}



        return sample
class CTRegressionModel(torch.nn.Module):
    def __init__(self):
        super(CTRegressionModel, self).__init__()
        # self.densenet = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)
        # self.densenet = EfficientNet(spatial_dims=3, in_channels=1, num_classes=1,blocks_args_str=["r1_k3_s11_e1_i32_o16_se0.25"])
        self.densenet = resnet50(spatial_dims=3, n_input_channels=1, num_classes=1,pretrained=False)
        state_dict = torch.load('/home/ubuntu/works/code/working_proj/OpenChallenge/pretrain/resnet_50_23dataset.pth')["state_dict"]
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}


        self.densenet.load_state_dict(new_state_dict,strict=False)

        self.fc = torch.nn.Linear(1 + 6, 1)  # 4额外特征 + 1来自DenseNet
        # self.fc = torch.nn.Linear(1, 1)  # 4额外特征 + 1来自DenseNet

    def forward(self, x, features):
        x = self.densenet(x)
        x = torch.cat([x, features], dim=1)
        x = self.fc(x)
        return x
def main():
    image_transforms = transforms.Compose([
        # transforms.ToPILImage(),  # 将numpy数组或torch张量转换为PIL图像
        # transforms.Resize((128, 128, 64)),  # 调整图像大小
        # transforms.ToTensor(),  # 将PIL图像转换为torch张量
        transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
    ])
    image_files = glob.glob('/media/ubuntu/disk/dataset/MEDdataset/chaimeleon/*/*/*.nii.gz')
    json_filesgt = []
    json_files_other = []
    wandb.init(project="chaimeleon", entity="baibizhe",name = 'only use age，geneder smoking status with resnet_50_23dataset crop 384^3 resize 128*128*64')
    batch_size=4

    wandb.config = {
        "batch_size": batch_size,
    }
    for i in image_files:
        gt = i.replace('.nii.gz','_ground_truth.json')
        json_filesgt.append(gt)
        other = i.replace('.nii.gz','.json')
        json_files_other.append(other)
        assert os.path.exists(gt),f'{gt} does not exist'
        assert  os.path.exists(other),f'{other} does not exist'
    assert  len(image_files) == len(json_files_other) == len(json_filesgt)

    print(f'length of images {len(image_files)} , length of json files {len(json_filesgt)} , {len(json_files_other)}  ')
    split_index = int(len(image_files)*0.8)
    train_dataset = CTImageDataset(image_files[0:split_index], json_filesgt[0:split_index],json_files_other[0:split_index], image_transforms)
    # 创建数据集
    val_dataset = CTImageDataset(image_files[split_index:], json_filesgt[split_index:],json_files_other[split_index:], image_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)

    # 创建模型、优化器和损失函数
    model = CTRegressionModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = MSELoss()
    num_epochs = 50
    # 训练循环
    # val_loader=train_loader=dataloader
    pbar = tqdm.tqdm(range(num_epochs))
    best_metrics = 99999
    for epoch in pbar:
        train_loss = 0
        model.train()
        train_loader = train_dataloader
        train_count = 0
        if epoch ==0:
            train_bar = tqdm.tqdm(train_loader)
        else:
            train_bar = train_loader

        for batch in train_bar:
            images = batch['image'].float().cuda()
            features = batch['features'].cuda()
            labels = batch['label'].float().cuda()

            # 前向传播
            outputs = model(images, features).squeeze(1)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_count+=images.shape[0]
            train_loss += loss.item()

            # break
        train_loss /= train_count
        wandb.log({"train_loss": loss.item()})

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():  # 在验证阶段不计算梯度
            for batch in val_dataloader:
                images = batch['image'].cuda().float()
                features = batch['features'].cuda()
                labels = batch['label'].cuda().float()

                # 前向传播
                outputs = model(images, features).squeeze(1)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_count += images.shape[0]
                wandb.log({"val_loss": loss.item()})

                # break

        # 平均验证损失
        val_loss /= val_count
        wandb.log({"epoch": epoch, "average_train_loss": train_loss, "average_val_loss": val_loss})

        if val_loss < best_metrics:
            best_metrics = val_loss
            torch.save(model,f'/home/ubuntu/works/code/working_proj/OpenChallenge/LungCancerOSPrediction/output/{epoch}.pt')
        # best_metrics = 0

        # 打印训练和验证损失
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f} , Best loss {best_metrics}")

if __name__ == '__main__':
    main()

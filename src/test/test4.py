from src.utils.image_process import *
from src.model.model import *
from src.process.task2_data_loader import get_dataloader
from src.process.task2_sliding_window2 import get_dataloader

# 把dataset放到DataLoader中
test_loader = get_dataloader(False)
model = UnetModel(1, 16, 6)
# model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'model_onehot2.pth')))
# model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'model_onehot.pth')))
# model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'Unet-final.pth')))
model.cpu()
model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save_task2', 'Unet-220.pth')))


def show_result(model):
    # 获取所有的valid样本
    test_data = data_set(False)
    data_loader = DataLoader(
        dataset=test_data,
        batch_size=1,
        pin_memory=True,
        shuffle=True
    )
    model.cpu()
    with torch.no_grad():
        # 对每一个测试案例做展示并保存
        for index, (x, y) in enumerate(data_loader):
            PRED = model(x.cpu().float())
            result = torch.argmax(PRED, dim=1)
            result = result.data.squeeze().cpu().numpy()
            save_image_information(index, result)
            pass


def show_result():
    # load data
    origin_image_path = '../utils/amos_0008.nii.gz'
    label_image_path = '../utils/amos_0008I.nii.gz'
    origin_image = sitk.GetArrayFromImage(sitk.ReadImage(origin_image_path))
    label_image = sitk.GetArrayFromImage(sitk.ReadImage(label_image_path))
    # origin image to [0,1] and mul 255
    origin_image = norm(origin_image) * 255
    # label image [0,15] with integer number

    # show_two
    pass


for i, j in test_loader:
    k = model(i.float())
    k = torch.argmax(k, 1)
    # bind(j, k)
    show_two(j, k, 'e-3', 2 / 3)
    pass
# show_result(model)

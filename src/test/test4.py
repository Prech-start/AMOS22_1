from src.utils.image_process import *
from src.model.model import *

test_data = data_set()
# 把dataset放到DataLoader中
test_loader = DataLoader(
    dataset=test_data,
    batch_size=1,
    shuffle=False
)
model = UnetModel(1, 16, 6)
# model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'model_onehot2.pth')))
# model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'model_onehot.pth')))
model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'Unet-final.pth')))
model.cpu()

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

for i, j in test_loader:
    k = model(i.float())
    k = torch.argmax(k, 1)
    # bind(j, k)
    show_two(j, k, 'e-3', 2 / 3)
    pass
# show_result(model)

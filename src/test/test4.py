from src.utils.image_process import *
from src.model.model import *

test_data = data_set()
# 把dataset放到DataLoader中
test_loader = DataLoader(
    dataset=test_data,
    batch_size=1,
    pin_memory=True,
    shuffle=False
)
model = UnetModel(1, 16, 6)
# model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'model_onehot2.pth')))
# model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'model_onehot.pth')))
model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'Generalized_Dice_loss_e-3_0.pth')))

for i, j in test_loader:
    k = model(i.float())
    k = torch.argmax(k, 1)
    # bind(j, k)
    show_two(j, k, 'e-3', 2 / 3)
    pass

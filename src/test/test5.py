from src.utils.image_process import *
from src.model.model import *

test_data = data_set()
# 把dataset放到DataLoader中
test_loader = DataLoader(
    dataset=test_data,
    batch_size=1,
    shuffle=False
)
# model = UnetModel(1, 16, 6)
# model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'model_onehot2.pth')))
# model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'model_onehot.pth')))
# model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'Unet-210.pth')))
# model.cpu()

# a = torch.Tensor(np.random.randint(0, 15,(1, 1, 64, WIDTH, HEIGHT)))
# concat_image(a, a, a)
if __name__ == '__main__':
    path = os.path.join('..', '..', 'data', 'AMOS22')
    with open(os.path.join('..', 'checkpoints', 'tr_ts_inf', 'testx.li_x.li'), 'rb+') as f:
        image_path = pickle.load(f)
    with open(os.path.join('..', 'checkpoints', 'tr_ts_inf', 'testx.li_y.li'), 'rb+') as f:
        gt_path = pickle.load(f)
    model = UnetModel(1, 16, 6)
    model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'Unet-210.pth')))
    for i, file_path in enumerate(image_path):
        # print(file_path)
        ori_image = sitk.ReadImage(os.path.join(path, bytes.decode(file_path)))
        GT = sitk.ReadImage(os.path.join(path, bytes.decode(gt_path[i])))
        model.cpu()
        x = copy.deepcopy(ori_image)
        x = np.array(sitk.GetArrayFromImage(x))
        GT = np.array(sitk.GetArrayFromImage(GT))
        x = resize(x, (64, 256, 256), order=1, preserve_range=True, anti_aliasing=False)
        GT = resize(GT, (64, 256, 256), order=0, preserve_range=True, anti_aliasing=False)
        q = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
        x = test_data.norm(q)
        GT = torch.from_numpy(GT.astype(np.int32)).type(torch.FloatTensor)
        pred = model(x)
        pred = torch.argmax(pred, dim=1)
        concat_image(q, GT, pred, no=1, slices=2.0 / 3)
        print(0)

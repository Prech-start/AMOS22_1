## 1.Dataset
You can download the datasets from [link](https://amos22.grand-challenge.org/) and idecompress it to the root directory.

## 2.Enviroment
- Please preparre an enviroment with python=3.7, and then use the command "pip install -r requirements.txt" to install the package which is necessary for running code correctly in this repository
- You should also own least one 2080ti-level gpu in hardware.

## 3.Run the whole work
- To train the model, If You want to train the latested model automatically, can use the following commands:
```bash
 python main.py
```
- To train the others model, You can type the another bash commands:
```bash
 python src/train/xx_train_yy.py
```
- where the xx means train strategy and the yy represents whether the model is a test_model or not, and is represented by test or blank.
- After training the model, the best and The best and latest trained models will be saved in [src/checkpoint/'strategy'/*](https://github.com/Prech-start/AMOS22_1/tree/master/src/checkpoints),which are named with 'Unet-final' and 'Unet-new'.

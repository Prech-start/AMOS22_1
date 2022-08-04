# if __name__ == '__main__':
#     for i in range(16):
#         print(i)

import easyocr
reader = easyocr.Reader(['ch_sim','en']) # 只需要运行一次就可以将模型加载到内存中
result = reader.readtext('chinese.jpg')

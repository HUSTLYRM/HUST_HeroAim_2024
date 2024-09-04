'''
@Email: 3284799532@qq.com
@Author: Deng Zehui
@Github: nArrow4
@Date: 2021-09-07 16:48:27
'''

import numpy as np
import cv2
# import joblib
# from sklearn.externals import joblib

isLoad = False

def load_module(path):
    svm = joblib.load(path)
    return svm

def svm_predict(armor):
    # if not isLoad:
    svm = load_module('/home/zhiyu/AutoAim/src/utils/tools/numbers.m')
    # isLoad = True
    gray = cv2.cvtColor(armor, cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(gray)
    # s = time.time()
    y_pred = svm.predict([np.array(hist.flatten())])
    # e = time.time()
    # print((e-s)*1000)
    return int(y_pred)
    pass

if __name__ == '__main__':

    import os
    import numpy as np
    import cv2
    from sklearn.externals import joblib
    model_root = '/home/zhiyu/AutoAim/src/utils/tools'
    model_path = os.path.join(model_root, "numbers.m")

    pic_root = '/home/zhiyu/AutoAim/src/utils/data/pictures/red_armor/'
    for i in range(10):
        pic_path = os.path.join(pic_root, str(i) + '.png')

        img = cv2.imread(pic_path)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # hist = cv2.equalizeHist(gray)

        y_pred = svm_predict(img)
        print(type(y_pred), int(y_pred))
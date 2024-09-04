'''
@Email: 3284799532@qq.com
@Author: Deng Zehui
@Github: nArrow4
@Date: 2021-09-12 11:48:08
'''
import cv2

svm = cv2.ml.SVM_load('src/utils/tools/svm_numbers.xml')

import os
import numpy as np

pic_root = '/home/zhiyu/AutoAim/src/utils/data/pictures/red_armor'
classes = ["0", "1", "2", "3", "4", "5"]

data_image = []
data_label = []

for class_ in classes:
    dir_ = os.path.join(pic_root, str(class_))
    # print(dir_)
    # data_label.extend([class_ for i in range(len(os.listdir(dir_)))])
    for file in os.listdir(dir_):
        # print(file)
        image = cv2.imread(os.path.join(dir_, file))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.equalizeHist(gray)
        res = hist.flatten()
        if len(res) == 1024:
            # data_image.append(np.array(res.flatten()))
            # data_label.append(np.array(class_))
            data_image.append(res.flatten())
            data_label.append(float(class_))
        # cv2.imshow("frame", hist)
        # cv2.waitKey(0)
data_image = np.array(data_image, dtype=np.float32)
data_label = np.array(data_label)





x_test = data_image
y_pred = svm.predict(x_test)[1]
y_test = data_label

# print(y_test)
# print(y_pred)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred, target_names=classes))
print(confusion_matrix(y_test, y_pred))
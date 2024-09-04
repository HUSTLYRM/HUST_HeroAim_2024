'''
@Email: 3284799532@qq.com
@Author: Deng Zehui
@Github: nArrow4
@Date: 2021-09-05 12:04:36
'''
import cv2
import os
import numpy as np

pic_root = '/home/zhiyu/AutoAim/src/utils/data/pictures/red_armor'
classes = ["1", "2", "3", "4", "5"]

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
            data_label.append(class_)
        # cv2.imshow("frame", hist)
        # cv2.waitKey(0)
data_image = np.array(data_image, dtype=np.float32)
data_label = np.array(data_label)
print(data_image.shape)
print(data_label.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_image, data_label, test_size=0.3, random_state=5, shuffle=True)
x_train = data_image
y_train = data_label

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setGamma(3)
svm.setDegree(3)
svm.setTermCriteria((cv2.TermCriteria_MAX_ITER, 300, 1e-3))
svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

y_pred = svm.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred, target_names=classes))
print(confusion_matrix(y_test, y_pred))
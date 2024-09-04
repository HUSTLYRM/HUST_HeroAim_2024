'''
@Email: 3284799532@qq.com
@Author: Deng Zehui
@Github: nArrow4
@Date: 2021-08-29 12:11:28
'''
import cv2
import os
import numpy as np

pic_root = '/home/zhiyu/AutoAim/src/utils/data/pictures/red_armor/'
classes = ["1", "2", "3", "4", "5"]

data_image = []
data_label = []

for class_ in classes:
    dir_ = os.path.join(pic_root + str(class_))
    # print(dir_)
    # data_label.extend([class_ for i in range(len(os.listdir(dir_)))])
    for file in os.listdir(dir_):
        # print(file)
        image = cv2.imread(os.path.join(dir_, file))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.equalizeHist(gray)
        res = hist.flatten()
        if len(res) == 1024:
            data_image.append(np.array(res.flatten()))
            data_label.append(np.array(class_))
        # cv2.imshow("frame", hist)
        # cv2.waitKey(0)
data_image = np.array(data_image)
data_label = np.array(data_label)
print(data_image.shape)
print(data_label.shape)

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(data_image, data_label, test_size=0.3, random_state=5, shuffle=True)

pca = PCA(n_components=150,whiten=True, random_state=42)
svc = SVC(kernel='rbf',class_weight='balanced')
pipeline = make_pipeline(pca, svc)

param_grid = {'svc__C': [1, 5, 10, 0.1],
                'svc__gamma': [0.0001, 0.0005, 0.001, 0.05]}
grid = GridSearchCV(pipeline, param_grid)
grid.fit(x_train, y_train)
print(grid.best_params_)

model = grid.best_estimator_
y_fit = model.predict(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test, y_fit, 
                            target_names=classes))
print(confusion_matrix(y_test, y_fit))

from sklearn.externals import joblib

joblib.dump(model, "src/utils/tools/numbers.m")

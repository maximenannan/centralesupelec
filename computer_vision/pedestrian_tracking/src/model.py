import os
import pickle
from typing import List
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC


def random_extract(img: np.ndarray, nb: int, width: int , height: int) -> List[np.ndarray]:
    """This function randomly extracts nb images of size (width x height) from img"""
    x_max_sample = img.shape[0] - height
    y_max_sample = img.shape[1] - width
    images = []
    for _ in range(nb):
        x = np.random.randint(0, x_max_sample + 1)
        y = np.random.randint(0, y_max_sample + 1)
        images.append(img[x: x+height, y:y+width])
    return images


def build_inria_dataset(
        positive_dir: str, negative_dir: str, dataset: str
) -> Tuple[List, List, List]:
    """
    This function build inria dataset that can be used to train a SVM to detect pedestrians
    within an image.
    You can find it there:
    http://lear.inrialpes.fr/data
    :param positive_dir: directory containing positive examples
    :param negative_dir: directory containing negative examples
    :param dataset: any of train or test (because both require a specific processing)
    :return:
    """
    images = []
    labels = []
    hogs = []
    if dataset == 'train':
        ymin, ymax, xmin, xmax = 16, 144, 16, 80
    elif dataset == 'test':
        ymin, ymax, xmin, xmax = 3, 131, 3, 67
    else:
        raise RuntimeError(f'unknown dataset {dataset}')
    hog = cv2.HOGDescriptor()
    # Extract positive samples
    for file in os.listdir(positive_dir):
        img = np.array(Image.open(os.path.join(positive_dir, file)))[ymin:ymax, xmin:xmax]
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2RGB)
        images.append(img)
        hogs.append(hog.compute(img).reshape(-1))
        labels.append(1)

    # Add negative samples
    for file in os.listdir(negative_dir):
        img = np.array(Image.open(os.path.join(negative_dir, file)))
        resized_image = cv2.resize(img, (64, 128))
        images.append(resized_image)
        hogs.append(hog.compute(resized_image).reshape(-1))
        labels.append(0)

        for sample_img in random_extract(img, 10, 64, 128):
            images.append(sample_img)
            hogs.append(hog.compute(sample_img).reshape(-1))
            labels.append(0)

    return images, labels, hogs


class Model(SVC):
    """
    This model is based on the model described in
    'Histograms of Oriented Gradients for Human Detection'
    """
    def __init__(self):
        super(Model, self).__init__(gamma='auto')
        self.hog = cv2.HOGDescriptor()

    def train(self, positive_dir: str, negative_dir: str, test_positive_dir: str = None,
              test_negative_dir: str = None):

        images, labels, hogs = build_inria_dataset(positive_dir, negative_dir, 'train')
        X_train = np.vstack(hogs)
        y_train = np.array(labels)

        self.fit(X_train, y_train)

        if test_positive_dir is not None and test_negative_dir is not None:
            _, test_labels, test_hogs = build_inria_dataset(
                test_positive_dir, test_negative_dir, 'test'
            )
            X_test = np.vstack(test_hogs)

            y_test = np.array(test_labels)

            y_pred = self.predict(X_test)

            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy = {accuracy} and f1 = {f1} on test set')

    def save_model(self, path):
        self.hog = None
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path):
        with open(path, 'rb') as f:
            clf = pickle.load(f)
        return clf

    def _predict(self, img):
        if self.hog is None:
            self.hog = cv2.HOGDescriptor()
        _W = 64
        _H = 128
        hog_features = self.hog.compute(cv2.resize(img, (_W, _H))).reshape(1, -1)
        return self.predict(hog_features)

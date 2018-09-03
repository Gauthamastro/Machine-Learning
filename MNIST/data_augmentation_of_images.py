from scipy.ndimage.interpolation import shift
import pickle
import numpy as np

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


label = pickle.load(open("label.list","rb"))
target = pickle.load(open("target.matrixlist","rb"))

X_train_augmented = [image for image in target]
y_train_augmented = [j for j in label]
print(len(X_train_augmented))
print(len(y_train_augmented))

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, j in zip(target, label):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(j)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)
pickle.dump(X_train_augmented,open("augmented.data","wb"))
pickle.dump(y_train_augmented,open("augmented.labels","wb"))


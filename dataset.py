import numpy as np


def load_morph_data_npz(npz_path, test_ratio = 0.2):
    f = np.load(npz_path)
    image, gender, age= f["image"], f["gender"], f["age"]
    image_shape0 = f["img_size"]
    f.close()

    data_num = len(image)
    train_num = int(data_num * (1 - test_ratio))

    image_train, gender_train, age_train = image[:train_num], gender[:train_num], age[:train_num]
    image_test, gender_test, age_test = image[train_num:], gender[train_num:], age[train_num:]

    return (image_train, age_train, gender_train), (image_test, age_test, gender_test), image_shape0


def load_megaage_data_npz(npz_path, test_ratio = 0.2):
    f = np.load(npz_path)
    image, age = f["image"], f["age"]
    image_shape0 = f["img_size"]
    f.close()

    data_num = len(image)
    train_num = int(data_num * (1 - test_ratio))

    image_train, age_train = image[:train_num], age[:train_num]
    image_test, age_test = image[train_num:], age[train_num:]

    return (image_train, age_train), (image_test, age_test), image_shape0

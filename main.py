from get_model import get_model
from random import shuffle
import os
from glob import glob


if __name__ == "__main__":
    model = get_model(yaml_path='setting.yaml')

    imagelist = sorted(glob(f"{model.image_path}/**/*.jpg", recursive=True))
    annotlist = sorted(glob(f"{model.annot_path}/**/*.png", recursive=True))
    assert len(imagelist) == len(annotlist), f"Image count {len(imagelist)} and annotation count {len(annotlist)} is different!"

    my_dataset = []
    for i, a in zip(imagelist, annotlist):
        my_dataset.append([i, a])

    count = len(my_dataset)
    shuffle(my_dataset)

    train_rate, valid_rate, test_rate = 0.8, 0.1, 0.1

    train_bundle = my_dataset[:int(count * train_rate)]
    valid_bundle = my_dataset[int(count * train_rate):int(count * (train_rate + valid_rate))]
    test_bundle = my_dataset[int(count * (train_rate + valid_rate)):]

    print("DATASET --- Train: {} | Valid: {} | Test: {}".format(len(train_bundle), len(valid_bundle), len(test_bundle)))

    trainset, validset, testset = [[], []], [[], []], [[], []]          # [ [image-list], [label-list] ]

    for elem in train_bundle:
        trainset[0].append(elem[0])
        trainset[1].append(elem[1])

    for elem in valid_bundle:
        validset[0].append(elem[0])
        validset[1].append(elem[1])

    for elem in test_bundle:
        testset[0].append(elem[0])
        testset[1].append(elem[1])

    model.fit(x=trainset[0],
              y=trainset[1],
              validation_data=validset,
              epochs=30,
              batch_size=4)
import argparse
import csv
import os

import numpy as np
from matplotlib import image


def convert_raw_csv(img_filepath, label_filepath, out_filepath, n_total):
    # original code form http://pjreddie.com/projects/mnist-in-csv/
    # The format of the resulting .csv files is:
    # label, pix-11, pix-12, pix-13, ..
    img_file = open(img_filepath, "rb")
    out_file = open(out_filepath, "w")
    label_file = open(label_filepath, "rb")

    img_file.read(16)
    label_file.read(8)
    images = []

    for ind_img in range(n_total):
        image = [ord(label_file.read(1))]
        for ind_pix in range(28 * 28):
            image.append(ord(img_file.read(1)))
        images.append(image)

    for image in images:
        out_file.write(",".join(str(pix) for pix in image) + "\n")
    img_file.close()
    out_file.close()
    label_file.close()


def convert_csv_npy(csv_filepath, out_img_filepath, out_labels_filepath, n_total):
    with open(csv_filepath, newline='') as csv_file:
        csv_read = csv.reader(csv_file)

        img_array = np.zeros((n_total, 28 * 28), dtype="uint8")
        label_array = np.zeros(n_total, dtype="uint8")
        for ind_row, row in enumerate(csv_read):
            ints = [int(num) for num in row]
            label_array[ind_row] = ints[0]
            img_array[ind_row, :] = ints[1:]
        np.save(out_img_filepath, img_array)
        np.save(out_labels_filepath, label_array)


def convert_npy_hdf5(img_filepath, labels_filepath, out_filepath):
    pass


def convert_npy_pngs(mnist_npy, dest_folder):
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
    for row_ind, row_vec in enumerate(mnist_npy):
        savepath = os.path.join(dest_folder, "IM" + str(row_ind).zfill(5) + ".png")
        image.imsave(savepath, row_vec.reshape((28, 28)), cmap="gray")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", action="store_true")
    parser.add_argument("-n", "--npy", action="store_true")
    parser.add_argument("-p", "--png", action="store_true")
    args = parser.parse_args()

    if args.csv:
        base_files = ["train-images.idx3-ubyte",
                      "train-labels.idx1-ubyte",
                      "t10k-images.idx3-ubyte",
                      "t10k-labels.idx1-ubyte"]
        if not all(os.path.exists(file) for file in base_files):
            base_files = ["train-images-idx3-ubyte",
                          "train-labels-idx1-ubyte",
                          "t10k-images-idx3-ubyte",
                          "t10k-labels-idx1-ubyte"]
            if not all(os.path.exists(file) for file in base_files):
                raise ValueError("One or more MNIST base files are missing. "
                                 "Is this script in the same directory as the "
                                 "files?")

        print("Converting to .csv...")
        print("Training data...")
        convert_raw_csv(base_files[0],
                        base_files[1],
                        "mnist_train.csv", 60000)
        print("Test data...")
        convert_raw_csv(base_files[2],
                        base_files[3],
                        "mnist_test.csv", 10000)
        print("...Done.")

    if args.npy:
        print("\nConverting to .npy...")
        print("Training data...")
        convert_csv_npy("mnist_train.csv", "mnist_train_imgs.npy",
                        "mnist_train_lbls.npy", 60000)
        print("Test data...")
        convert_csv_npy("mnist_test.csv", "mnist_test_imgs.npy",
                        "mnist_test_lbls.npy", 10000)
        print("...Done.")

    if args.png:
        print("\nConverting to images...")
        print("Training data...")
        convert_npy_pngs(np.load("mnist_train_imgs.npy"), "train")
        print("Test data...")
        convert_npy_pngs(np.load("mnist_test_imgs.npy"), "test")
        print("...Done.")

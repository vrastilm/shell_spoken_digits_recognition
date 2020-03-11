#!/usr/bin/env python
# -*- coding: utf-8 -*-

from preprocessing import wave_to_mfcc_matrix
import logging
import argparse
from pprint import pprint
import numpy as np
import os
from os.path import join, dirname, exists, abspath
import sys
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from mapping import get_one_hot_vec, get_number_of_classes

# set currrent file directory as working directory
os.chdir(dirname(abspath(__file__)))


def _create_logger():
    # Create a custom logger
    logger = logging.getLogger("shell_asr")
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('[%(asctime)s][%(levelname)s]: %(message)s')
    c_handler.setFormatter(c_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)

    return logger


LOGGER = _create_logger()


def _check_sample_rate(fs: int, fs_wave: int) -> None:
    if fs == 0:
        fs = fs_wave
    if fs != 0 and fs != fs_wave:
        ValueError(
            "Sample frequency is not consistent in dataset!")


def _prepare_google_commands_dataset_data(data_folder: str,
                                          max_wave_len: int,
                                          win_len: int,
                                          max_mfcc_coeffs: int,
                                          train_fnames: List[str],
                                          validation_fnames: List[str],
                                          ignored_names: Tuple[str] = ()) -> Tuple[np.ndarray, np.ndarray]:
    fs = 0
    data_list = np.array([])
    validation_list = np.array([])
    for root, _, files in os.walk(data_folder):
        for i, fn in enumerate(files):
            dir_name = root.split('/')[-1]
            if dir_name not in ignored_names:
                if fn not in ignored_names:
                    if join(dir_name, fn) not in ignored_names:
                        if join(dir_name, fn) in train_fnames:
                            filepath = join(root, fn)

                            if i % 100 == 0:
                                LOGGER.info(
                                    f"{i} files of {len(files)} in folder {dir_name} processed")
                            fs_wave, data_matrix = wave_to_mfcc_matrix(filepath,
                                                                       max_wave_len,
                                                                       win_len,
                                                                       max_mfcc_coeffs)
                            _check_sample_rate(fs, fs_wave)

                            dir_vector = get_one_hot_vec(
                                data_folder, dir_name, ignored_names)
                            data_list = np.append(data_list,
                                                  (dir_vector, data_matrix))

                        elif join(dir_name, fn) in validation_fnames:
                            if i % 100 == 0:
                                LOGGER.info(
                                    f"{i} files of {len(files)} in folder {dir_name} processed")

                            fs_wave, data_matrix = wave_to_mfcc_matrix(filepath,
                                                                       max_wave_len,
                                                                       win_len,
                                                                       max_mfcc_coeffs)
                            _check_sample_rate(fs, fs_wave)

                            dir_vector = get_one_hot_vec(
                                data_folder, dir_name, ignored_names)
                            data_list = np.append(validation_list,
                                                  (dir_vector, data_matrix))

    return data_list, validation_list


def _get_file_names_from_file(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r') as file:
            return [item.strip() for item in file.readlines()]
    except FileExistsError:
        LOGGER.error(f"File {file_path.split('/')[-2]} doesnt exist!")


def _data_list_split(data: List[Tuple[int, np.array]]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = ([], [])

    for i in range(0, len(data)-1, 2):
        y.append(data[i])
        X.append(data[i+1])

    return (np.array(X), np.array(y))


def _create_model(n_out: int) -> Sequential:
    # basic sequential model
    model = Sequential()

    model.add(Conv2D(32,
                     kernel_size=(4, 4),
                     strides=(2, 2),
                     activation='relu',
                     input_shape=(20, 79, 1)))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(2432, activation='relu'))

    model.add(Dense(1024, activation='relu'))

    model.add(Dense(n_out, activation='softmax'))

    return model


def _plot_model_history(history: any, filename: str) -> None:
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Val_accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig(filename)


def _get_file_names_shaked_and_splitted(data_folder: str,
                                        ignored_names: List[str],
                                        test_split: float = 0.2) -> Tuple[List[str], List[str]]:
    all_files = []
    for root, dirs, _ in os.walk(data_folder, topdown=False):
        for dirct in dirs:
            if dirct not in ignored_names:
                for filename in os.listdir(join(root, dirct)):
                    folder_file_name = join(dirct, filename)
                    if folder_file_name not in ignored_names:
                        all_files.append(folder_file_name)

    return train_test_split(all_files, test_size=test_split)


def main():
    LOGGER.info('--- Starting shell asr model training script ---')

    parser = argparse.ArgumentParser(
        description='Script for training the model for shell automatic speech recognition of google v2 commands dataset')
    parser.add_argument('-d', '--data_folder',
                        required=True,
                        default=None,
                        help='Google dataset root folder path')
    parser.add_argument('-s', '--train_split',
                        required=False,
                        default=0.8,
                        help='Train-test split of dataset in percentage')
    parser.add_argument('-l', '--max_wave_len',
                        required=False,
                        default=40000,
                        help='Maximum length of wave data array')
    parser.add_argument('-w', '--win_len',
                        required=False,
                        default=350,
                        help='Length of window')
    parser.add_argument('-m', '--mfcc_coeffs_len',
                        required=False,
                        default=20,
                        help='Number of mfcc spectral coefficients')
    parser.add_argument('-n', '--model_name',
                        required=False,
                        default='saved_model/shell_asr_model',
                        help='Name of model to be exported')
    parser.add_argument('-tdc', '--train_data_cache',
                        required=False,
                        default='cache/train_data.npy',
                        help='File for storing train data cache')
    parser.add_argument('-vdc', '--validation_data_cache',
                        required=False,
                        default='cache/validate_data.npy',
                        help='File for storing validation data cache')
    args = parser.parse_args()

    data_folder = args.data_folder
    train_dataset_split = float(args.train_split)
    max_wave_len = int(args.max_wave_len)
    win_len = int(args.win_len)
    max_mfcc_coeffs = int(args.mfcc_coeffs_len)

    # check correctness of input data
    assert exists(data_folder), 'Google dataset folder does not exists'
    assert 0 < train_dataset_split < 1, 'Train-test split should be float from interval (0,1)'
    assert 0 < max_wave_len <= 100000, 'Max wave array length should int be from interval (0,100000>'
    assert 0 < win_len <= 500, 'Max wave array length should be int from interval (0,500>'
    assert 0 < max_mfcc_coeffs <= 30, 'Number of mfcc coefficients should be int from interval (0, 30>'
    assert win_len < max_wave_len, 'Window length should be int less than wave data array length'
    assert len(args.model_name) > 0, 'Model name should not be empty string'

    LOGGER.warning('Loading data to memory...')

    # files to be ignored i data folder
    ignored_names = ('_background_noise_', 'LICENSE',
                     'README.md', 'testing_list.txt', 'validation_list.txt')

    # if no cache provided create new training data
    if exists(args.train_data_cache) and exists(args.validation_data_cache):
        LOGGER.warning('Using cached data...')
        train_data = np.load(args.train_data_cache, allow_pickle=True)

        validation_data = np.load(
            args.validation_data_cache, allow_pickle=True)
    else:
        train_fnames, validation_fnames = _get_file_names_shaked_and_splitted(
            data_folder, ignored_names, test_split=0.2)

        train_data, validation_data = _prepare_google_commands_dataset_data(
            data_folder,
            max_wave_len,
            win_len,
            max_mfcc_coeffs,
            train_fnames,
            validation_fnames,
            ignored_names=ignored_names
        )

        # cache computed values
        np.save(args.train_data_cache, train_data)
        np.save(args.validation_data_cache, validation_data)

        # little memory management
        del train_fnames
        del validation_fnames

    LOGGER.warning('Creating datasets...')

    X, y = _data_list_split(train_data)
    X = np.reshape(X, (X.shape[0], 20, 79, 1))
    X, y = shuffle(X, y)

    X_validation, y_validation_hot = _data_list_split(validation_data)
    X_validation = np.reshape(X_validation, (X_validation.shape[0], 20, 79, 1))
    X_validation, y_validation_hot = shuffle(X_validation, y_validation_hot)

    LOGGER.warning('Creating model...')

    model = _create_model(get_number_of_classes(data_folder, ignored_names))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.optimizers.Adadelta(),
                  metrics=['accuracy'])

    LOGGER.warning('Model training and testing...')

    model.summary()

    history = model.fit(X, y, batch_size=100, epochs=300,
                        validation_data=(X_validation, y_validation_hot))

    # print history figure
    filename = 'val_accuracy_over_epochs.png'
    LOGGER.warning(
        f"Printing accuracy and val accuracy figure to file: {filename}...")
    _plot_model_history(history, filename)

    # export model
    LOGGER.info(f"Exporting trained model: {args.model_name}...")
    model.save(args.model_name)


if __name__ == '__main__':
    main()

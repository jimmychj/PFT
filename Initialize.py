from pdf2image import convert_from_path
import imagehash
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
from itertools import zip_longest
import os


def init(directory):
    sampimg = Image.open(directory + 'Model.png')
    box = sampimg.crop((230, 253, 245, 267))
    boxhash = imagehash.average_hash(box)

    imlabel = Image.open(directory + 'imlabel.png')
    hashL = imagehash.average_hash(imlabel)

    number1 = Image.open(directory + 'number1.png')
    number2 = Image.open(directory + 'number2.png')
    number5 = Image.open(directory + 'number5.png')
    number0 = Image.open(directory + 'number0.png')
    number20 = Image.open(directory + 'number20.png')
    number10 = Image.open(directory + 'number10.png')
    number00 = Image.open(directory + 'number00.png')

    hash1 = imagehash.average_hash(number1)
    hash2 = imagehash.average_hash(number2)
    hash5 = imagehash.average_hash(number5)
    hash0 = imagehash.average_hash(number0)
    hash20 = imagehash.average_hash(number20)
    hash10 = imagehash.average_hash(number10)
    hash00 = imagehash.average_hash(number00)
    hash_data = [boxhash, hashL, hash2, hash0, hash20, hash10, hash00, hash1, hash5]
    return hash_data


if __name__ == '__main__':
    directory = 'C:/Users/hc258/PFT Project/sample/'
    a = init(directory)
    print(a)

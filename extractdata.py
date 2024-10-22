from pdf2image import convert_from_path
import imagehash
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
from itertools import zip_longest
import os
import time
import logging


def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i >= width or j >= height or i <= 0 or j <= 0:
        return 255, 255
    # Get Pixel
    pixel = image.getpixel((i, j))
    return pixel


def create_image(i, j):
    image = Image.new("RGB", (i, j), 'white')
    return image


def generate_path_and_name(directory, filename):
    path = os.path.join(directory, filename)
    string2 = directory
    string3 = '.Pdf'
    name = path.replace(string2, '')
    name = name.replace(string3, '')
    return path, name


def find_figure(image_pdf, search_range_x, search_range_y, offset, hash_l):
    # x = [1200, 1635]  y = [360, 600]  figure_size = [labelx'-300', labely'-50', '1700', labely'+500']
    image = image_pdf[0]
    label_x = []
    label_y = []
    for x in range(search_range_x[0], search_range_x[1]):
        for y in range(search_range_y[0], search_range_y[1]):
            section = image.crop((x, y, x + 55, y + 13))
            hash_target = imagehash.average_hash(section)
            if hash_l - hash_target < 10:
                label_x.append(x)
                label_y.append(y)
                break
        else:
            continue
        break
    label_x_avg = int(np.mean(label_x))
    label_y_avg = int(np.mean(label_y))
    section = image.crop((label_x_avg - offset[0], label_y_avg - offset[1],
                          offset[2], label_y_avg + offset[3]))
    w, h = section.size
    contour = create_image(w, h)
    pixels = contour.load()
    for x in range(w):
        for y in range(h):
            if x in range(offset[0], w) and y in range(offset[1]-20, offset[1]+20):
                pixels[x, y] = (255, 255, 255)
            else:
                pixels[x, y] = section.getpixel((x,y))
    # plt.imshow(contour)
    # plt.show()
    return contour


def find_2_in_axis(section, n2_offset, hash2, hash20, hash5):
    align_x = []
    align_y = []
    zero = 0
    for x in range(1, 300):
        for y in range(1, 300):
            maxy = section.crop((x, y, x + 9, y + 15))
            maxy0 = section.crop((x, y, x + 27, y + 15))
            hash_n = imagehash.average_hash(maxy)
            hash_n0 = imagehash.average_hash(maxy0)
            if hash2 - hash_n < 10 and zero == 0:
                if hash20 - hash_n0 < 10:
                    align_x.append(x + 17)
                    align_y.append(y)
                    print(x, y, '2.0')
                    c = 2
                    zero = 1
                else:
                    align_x.append(x)
                    align_y.append(y)
                    print(x, y, '2')
                    c = 2
                break
            elif hash5 - hash_n < 10:
                if len(align_x) > 0 and x - align_x[-1] > 5:
                    pass
                else:
                    align_x.append(x)
                    align_y.append(y)
                    c = 5
                    print(x, y, '5')
                    break
        else:
            continue
        break
    align_x = int(np.mean(align_x))
    align_y = int(np.mean(align_y))
    a_x = align_x + n2_offset[0]
    a_y = align_y + n2_offset[1]
    return a_x, a_y, c


def find_1_10_2_20_axis(section, offset, hash10, hash2, hash20, ay_x, ay_y, hash1):
    align_x = []
    align_y = []
    for x in range(ay_x + 10, 500):
        for y in range(ay_y+10, 380):
            max_x = section.crop((x, y, x + 9, y + 15))
            max_x0 = section.crop((x, y, x + 27, y + 15))
            hash_n = imagehash.average_hash(max_x)
            hash_n0 = imagehash.average_hash(max_x0)
            if hash1 - hash_n < 9:
                if hash10 - hash_n0 < 13:
                    if len(align_x) > 0 and x - align_x[-1] > 5:
                        pass
                    else:
                        align_x.append(x + 7)
                        align_y.append(y)
                        c = 1
                        print(x, y, '1.0')
                        break
                else:
                    if len(align_x) > 0 and x - align_x[-1] > 5:
                        pass
                    else:
                        align_x.append(x)
                        align_y.append(y)
                        c = 1
                        print(x, y, '1')
                        break
            elif hash2 - hash_n < 10:
                if hash20 - hash_n0 < 13:
                    if len(align_x) > 0 and x - align_x[-1] > 5:
                        pass
                    else:
                        align_x.append(x + 7)
                        align_y.append(y)
                        c = 2
                        print(x, y, '2.0')
                        break
                else:
                    if len(align_x) > 0 and x - align_x[-1] > 5:
                        pass
                    else:
                        align_x.append(x)
                        align_y.append(y)
                        c = 2
                        print(x, y, '2')
                        break
        else:
            continue
        break
    align_x = int(np.mean(align_x))
    align_y = int(np.mean(align_y))
    ax_x = align_x + offset[0]
    ax_y = align_y + offset[1]
    return ax_x, ax_y, c


def process_color(image_test, color_code):
    w, h = image_test.size
    contour = create_image(w, h)
    pixels = contour.load()
    threshold = 180
    # plt.imshow(image_test)
    for x in range(w):
        for y in range(h):
            if color_code == 0:
                color = get_pixel(image_test, x, y)
                if color[0] <= threshold <= color[2] and color[1] <= threshold:
                    pixels[x, y] = color
                else:
                    pixels[x, y] = (255, 255, 255)
    # plt.figure()
    # plt.imshow(contour)
    # plt.show()
    return contour


def process_color_bw(image_test):
    w, h = image_test.size
    contour = create_image(w, h)
    pixels = contour.load()
    threshold = 180
    plt.imshow(image_test)
    for x in range(w):
        for y in range(h):
            color = get_pixel(image_test, x, y)
            if len(color)==3:
                if color[2]>color[0] and color[2]>color[1]:
                    pixels[x, y] = (255, 255, 255)
                else:
                    pixels[x, y] = color
            else:
                pixels[x, y] = (255, 255, 255)
    return contour


def detect_box(im, boxhash):
    boxx = []
    boxy = []
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            corner = im.crop((x, y, x + 15, y + 14))
            chash = imagehash.average_hash(corner)
            if abs(chash - boxhash) < 12:
                boxx.append(x)
                boxy.append(y)
                break
    if boxx == []:
        print('1')
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                corner = im.crop((x, y, x + 15, y + 14))
                chash = imagehash.average_hash(corner)
                if abs(chash - boxhash) < 15:
                    boxx.append(x)
                    boxy.append(y)
                    break
    if boxx == []:
        print('2')
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                corner = im.crop((x, y, x + 15, y + 14))
                chash = imagehash.average_hash(corner)
                if abs(chash - boxhash) < 18:
                    boxx.append(x)
                    boxy.append(y)
                    break
    return boxx, boxy


def remove_box(im, boxx, boxy):
    test_img = im.convert('LA')
    pix = test_img.load()
    box_x = int(np.mean(boxx))
    box_y = int(np.mean(boxy))
    for x in range(box_x - 2, box_x + 17):
        for y in range(box_y - 30, box_y + 10):
            pix[x, y] = (255, 255)
    return test_img


def point_ext(im, ay_x, ax_y):
    width = im.size[0]
    height = im.size[1]
    point_x = []
    point_y = []
    for w in range(0, width):
        for h in range(0, ax_y):
            color = get_pixel(im, w, h)
            if color[0] < 100:
                point_x.append(w)
                point_y.append(h)
    return point_x, point_y


def find_peak_end(test_img, ay_x, ax_y):
    point_x, point_y = point_ext(test_img, ay_x, ax_y)
    x_min = max(point_x)
    y_min = point_y[np.argmax(point_x)]
    y_peak = min(point_y)
    x_peak = point_x[np.argmin(point_y)]
    return x_min, y_min, x_peak, y_peak


def curve_ext_down(test_img, threshold, x_min, y_min, x_peak, y_peak):
    # from peak to end
    pointx = [x_peak]
    pointy = [y_peak]
    for i in range(600):
        for dx in range(1, 10):
            for dy in range(-10, 30):
                x = pointx[-1] + dx
                y = pointy[-1] + dy
                color = get_pixel(test_img, x, y)
                if color[0] < threshold and x < x_min:
                    pointx.append(x)
                    pointy.append(y)
                    break
            else:
                continue
            break
    return pointx, pointy


def curve_ext_end(test_img, threshold, x_min, y_min, x_peak, y_peak, box_x, box_y):
    pointx = [x_min]
    pointy = [y_min]
    ratio = (y_min - box_y) / (x_min - box_x)
    for i in range(150):
        for dx in range(20):
            for t in range(5):
                dy_l = ratio * (-2) * dx
                dy_u = ratio * 1.5 * t * dx
                for dy in range(int(dy_l), int(dy_u)):
                    x = pointx[-1] - dx
                    y = pointy[-1] - dy
                    color = get_pixel(test_img, x, y)
                    if color[0] < threshold and x > box_x + 10:
                        pointx.append(x)
                        pointy.append(y)
                        break
                else:
                    continue
                break
            else:
                continue
            break
    return pointx, pointy


def curve_ext_up(test_img, threshold, ay_x, ax_y, x_peak, y_peak):
    pointx = [0]
    pointy = [ax_y]
    ratio = (ax_y - y_peak) / (x_peak - 0)
    for i in range(150):
        for dx in range(20):
            for t in range(10):
                dy_l = ratio * (-1.5) * dx
                dy_u = ratio * 1.5 * t * dx
                for dy in reversed(range(int(dy_l), int(dy_u))):
                    x = pointx[-1] + dx
                    y = pointy[-1] - dy
                    color = get_pixel(test_img, x, y)
                    if color[0] < threshold and x < x_peak:
                        pointx.append(x)
                        pointy.append(y)
                        break
                else:
                    continue
                break
            else:
                continue
            break
    return pointx, pointy


def extract_data(directory, hash_data, save_dir):
    box_hash = hash_data[0]
    hash_l = hash_data[1]
    hash2 = hash_data[2]
    hash0 = hash_data[3]
    hash20 = hash_data[4]
    hash10 = hash_data[5]
    hash00 = hash_data[6]
    hash1 = hash_data[7]
    hash5 = hash_data[8]
    for filename in os.listdir(directory):
        if filename.endswith(".Pdf"):
            start = time.time()
            try:
                # Load pdf and find the figure
                path, name = generate_path_and_name(directory, filename)
                print(name)
                image_pdf = convert_from_path(path)  #, poppler_path='C:/Users/ch/Desktop/Release-22.04.0-0/poppler-22.04.0/Library/bin')
                search_range_x = [1200, 1635]  # [200, 500] for Raleigh, [1200, 1635] for Durham
                search_range_y = [360, 600] # [1400, 2200] for Raleigh, [360, 600] for Durham
                offset = [300, 50, 1700, 500] # [250, 30, 600, 300] for Raleigh, [300, 50, 1700, 500] for Durham
                print("Starting to find figure...")
                section = find_figure(image_pdf, search_range_x, search_range_y, offset, hash_l)
                section_processed = process_color_bw(section)

                # Alignment for y-axis
                print("Starting to make alignment...")
                n2_offset_y = [21, 7]
                ay_x, ay_y, c_y = find_2_in_axis(section_processed, n2_offset_y, hash2, hash20, hash5)
                print("Finished Y-Axis Alignment")
                # plt.imshow(section)
                # plt.scatter(ay_x, ay_y)
                # plt.show()

                # x-axis
                n2_offset_x = [5, -15]
                ax_x, ax_y, number = find_1_10_2_20_axis(section_processed, n2_offset_x, hash10, hash2, hash20, ay_x, ay_y, hash1)
                print("Finished X-Axis Alignment")
                # plt.imshow(section)
                # plt.scatter(ax_x, ax_y)
                # plt.show()

                # Calculating mapping coefficient on x and y-axis
                map_y = c_y / abs(ay_y - ax_y)
                map_x = number / abs(ax_x - ay_x)
                w, h = section.size
                contour = process_color(section, 0)
                im = contour.crop((ay_x, 0, w, ax_y))
                # boxx, boxy = detect_box(im, box_hash)
                # box_x = int(np.mean(boxx))
                # box_y = int(np.mean(boxy))
                test_img = im.convert('LA')
                # plt.imshow(test_img)
                # plt.show()

                # Find the peak point and the end point
                x_min, y_min, x_peak, y_peak = find_peak_end(test_img, ay_x, ax_y)
                # Extract the curve
                point_x_up, point_y_up = curve_ext_up(test_img, 200, ay_x, ax_y, x_peak, y_peak)
                point_x_pb, point_y_pb = curve_ext_down(test_img, 200, x_min, y_min, x_peak, y_peak)
                # point_x_be, point_y_be = curve_ext_end(test_img, 200, x_min, y_min, x_peak, y_peak, box_x, box_y)
                pointx = point_x_up + point_x_pb
                pointy = point_y_up + point_y_pb
                cordx = map_x * (np.array(pointx) - 0)
                cordy = map_y * (ax_y - np.array(pointy))
                d = [cordx, cordy]
                export_data = zip_longest(*d, fillvalue='')

                # Check plotted data after alignment
                # plt.show()

                # writing the data into the file
                with open(save_dir + name + '.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
                    wr = csv.writer(myfile)
                    #     wr.writerow(("X", "Y"))
                    wr.writerows(export_data)
                myfile.close()

                from dataproc import import_data
                df = import_data(save_dir + name + '.csv')
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(section)
                plt.subplot(1, 2, 2)
                plt.scatter(df['x'], df['y'])
                plt.savefig(save_dir + name + '.png')
                plt.clf()
                end = time.time()
                print("This process took {} seconds".format(int(end - start)))
            except ValueError:
                print('The file name {} is problematic !!!!!!!!!'.format(name))
        else:
            continue

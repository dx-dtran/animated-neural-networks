import numpy as np
import matplotlib.pyplot as plt
import csv
import math

from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])


def get_xor_data(num_samples: int, noise=0):
    def get_xor_label(point: Point):
        return 1 if point.x * point.y >= 0 else -1

    padding = 0.3
    data = []
    labels = []
    for i in range(num_samples):
        x = np.random.uniform(-5, 5)
        x += padding if x > 0 else -padding

        y = np.random.uniform(-5, 5)
        y += padding if y > 0 else -padding

        noise_x = np.random.uniform(-5, 5) * noise
        noise_y = np.random.uniform(-5, 5) * noise

        label = get_xor_label(Point(x + noise_x, y + noise_y))

        data.append((x, y, 0.0))
        labels.append(label)

    return data, labels


def get_circle_data(num_samples: int, noise=0):
    data = []
    labels = []
    radius = 5

    def get_circle_label(point: Point, center: Point):
        return 1 if euclidian_distance(point, center) < (radius * 0.5) else -1

    for i in range(num_samples // 2):
        r = np.random.uniform(0, radius * 0.4)
        angle = np.random.uniform(0, 2 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        noise_x = np.random.uniform(-radius, radius) * noise
        noise_y = np.random.uniform(-radius, radius) * noise
        label = get_circle_label(Point(x + noise_x, y + noise_y), Point(0, 0))
        data.append((x, y, 0.0))
        labels.append(label)

    for i in range(num_samples // 2):
        r = np.random.uniform(0, radius * 0.8)
        angle = np.random.uniform(0, 2 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        noise_x = np.random.uniform(-radius, radius) * noise
        noise_y = np.random.uniform(-radius, radius) * noise
        label = get_circle_label(Point(x + noise_x, y + noise_y), Point(0, 0))
        data.append((x, y, 0.0))
        labels.append(label)

    return data, labels


def euclidian_distance(a: Point, b: Point):
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def get_class_groups(points, labels):
    groups = []
    for label in labels:
        groups.append(get_class_points(points, label))
    return groups


def get_class_points(points, label):
    x = np.array([pt[0] for pt in points if pt[2] == label])
    y = np.array([pt[1] for pt in points if pt[2] == label])
    return x, y


def write_data(filename, data):
    with open(filename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, lineterminator="\n")
        for row in data:
            csv_writer.writerow(row)


def read_2d_data(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        data = []
        labels = []
        for row in csv_reader:
            x, y, label = row
            data.append((float(x), float(y)))
            labels.append(float(label))
    return data, labels


def read_2d_data_in_3d(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        data = []
        labels = []
        for row in csv_reader:
            x, y, label = row
            data.append((float(x), float(y), 0.0))
            labels.append(float(label))
    return data, labels


def plot(data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    print(data)
    data = get_class_groups(data, [-1, 1])
    colors = ("orange", "blue")
    for data, color in zip(data, colors):
        x_val, y_val = data
        ax.scatter(x_val, y_val, alpha=0.8, c=color)

    plt.title("my scatter plot")
    plt.show()


def main():
    xor_data = get_xor_data(200)
    write_data("../data/xor_data.csv", xor_data)


if __name__ == "__main__":
    main()

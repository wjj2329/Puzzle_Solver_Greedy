import subprocess

import imageio.v3 as iio
from skimage import color

from .enums import ColorType, ScoreAlgorithm
from .image_io import ensureOutputDirectory, prepareImageForWrite
from .models import Segment
from .paths import IMAGE_OUTPUT_DIR


def get_gist(filename):
    data = open(filename, 'r').read()
    return [float(x) for x in data.split()]


def breakUpImage(
        image,
        length,
        save_segments,
        color_type,
        score_algorithm,
        output_dir=IMAGE_OUTPUT_DIR):
    dimensions = image.shape
    if dimensions[0] != dimensions[1]:
        print("Only square images will work for now to keep things simple")
        exit()
    if dimensions[0] % length != 0 or dimensions[1] % length != 0:
        print("unable to break up image into equal squares")
        exit()
    segments = []
    x, y = 0, 0
    pic_x, pic_y = 0, 0
    piece_num = 1
    num_of_pieces_width = int(dimensions[0]/length)
    num_of_pieces_height = int(dimensions[1]/length)
    append = segments.append
    score_dict = {}
    connections_dict = {}
    if save_segments:
        output_dir = ensureOutputDirectory(output_dir)
    for x in range(num_of_pieces_width):
        for y in range(num_of_pieces_height):
            save = image[pic_x: pic_x+length, pic_y: pic_y+length, :]
            gist = None
            if save_segments:
                image_path = output_dir / f"{x}_{y}.png"
                if color_type == ColorType.RGB:
                    iio.imwrite(image_path, prepareImageForWrite(save))
                elif color_type == ColorType.LAB:
                    image_temp = color.lab2rgb(save)
                    iio.imwrite(image_path, prepareImageForWrite(image_temp))
                elif score_algorithm == ScoreAlgorithm.GIST_AND_EUCLIDEAN:
                    subprocess.run(["gist.exe", "-i", str(image_path), "-o", str(output_dir)])
                    gist = get_gist(output_dir / "gist.txt")
            segment_to_append = Segment(save, num_of_pieces_width,
                                        num_of_pieces_height, piece_num, piece_num, score_dict, gist, connections_dict)
            append(segment_to_append)
            piece_num += 1
            pic_y += length
        pic_x += length
        pic_y = 0
    return segments

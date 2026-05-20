from pathlib import Path

import imageio.v3 as iio
import numpy as np
from numpy import zeros
from skimage import color

from .enums import ColorType
from .paths import IMAGE_OUTPUT_DIR


def prepareImageForWrite(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if image.size > 0 and image.min() >= 0 and image.max() <= 1:
            image = image * 255
        image = np.clip(image, 0, 255).round().astype(np.uint8)
    return image


def ensureOutputDirectory(output_dir=IMAGE_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def saveImage(
        best_connection,
        piece_size,
        round_number,
        color_type,
        name_for_round,
        output_dir=IMAGE_OUTPUT_DIR):
    pic_locations = best_connection.binary_connection_matrix.nonzero()
    biggestx = max(pic_locations[0])
    biggesty = max(pic_locations[1])
    smallestx = min(pic_locations[0])
    smallesty = min(pic_locations[1])
    sizex = (biggestx-smallestx)+1
    sizey = (biggesty-smallesty)+1
    biggest_dim = sizex if sizex > sizey else sizey
    new_image = zeros((biggest_dim*piece_size, biggest_dim*piece_size, 3))
    for x in range(len(pic_locations[0])):
        piece_to_assemble = best_connection.pic_connection_matrix[pic_locations[0]
                                                                 [x], pic_locations[1][x]].pic_matrix
        x1 = (pic_locations[0][x]-smallestx)*piece_size
        y1 = (pic_locations[1][x]-smallesty)*piece_size
        new_image[x1:x1+piece_size, y1:y1+piece_size, :] = piece_to_assemble
    if color_type == ColorType.LAB:
        new_image = color.lab2rgb(new_image)
    output_dir = ensureOutputDirectory(output_dir)
    image_path = output_dir / f"{name_for_round} round{round_number}.png"
    iio.imwrite(image_path, prepareImageForWrite(new_image))
    return str(image_path)

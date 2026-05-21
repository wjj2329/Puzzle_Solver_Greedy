import imageio.v3 as iio
from skimage import color

from .enums import ColorType
from .image_io import ensureOutputDirectory, prepareImageForWrite
from .models import Segment
from .paths import IMAGE_OUTPUT_DIR


def breakUpImage(
        image,
        length,
        save_segments,
        color_type,
        output_dir=IMAGE_OUTPUT_DIR):
    dimensions = image.shape
    if dimensions[0] != dimensions[1]:
        print("Only square images will work for now to keep things simple")
        raise SystemExit(1)
    if dimensions[0] % length != 0 or dimensions[1] % length != 0:
        print("unable to break up image into equal squares")
        raise SystemExit(1)
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
            if save_segments:
                image_path = output_dir / f"{x}_{y}.png"
                if color_type == ColorType.RGB:
                    iio.imwrite(image_path, prepareImageForWrite(save))
                elif color_type == ColorType.LAB:
                    image_temp = color.lab2rgb(save)
                    iio.imwrite(image_path, prepareImageForWrite(image_temp))
            segment_to_append = Segment(save, num_of_pieces_width,
                                        num_of_pieces_height, piece_num, piece_num, score_dict, connections_dict)
            append(segment_to_append)
            piece_num += 1
            pic_y += length
        pic_x += length
        pic_y = 0
    return segments

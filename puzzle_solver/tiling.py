from concurrent.futures import ThreadPoolExecutor

import imageio.v3 as iio
from skimage import color

from .enums import ColorType
from .image_io import ensureOutputDirectory, prepareImageForWrite
from .models import Segment
from .paths import IMAGE_OUTPUT_DIR


class SegmentSaveBatch:
    def __init__(self, executor, futures):
        self.executor = executor
        self.futures = futures

    def wait(self):
        try:
            for future in self.futures:
                future.result()
        finally:
            self.executor.shutdown(wait=True)


def writeSegmentImage(image_path, segment_image, color_type):
    if color_type == ColorType.LAB:
        segment_image = color.lab2rgb(segment_image)
    iio.imwrite(image_path, prepareImageForWrite(segment_image))


def segmentImagePath(segment, output_dir):
    piece_index = segment.piece_number - 1
    x = piece_index // segment.max_height
    y = piece_index % segment.max_height
    return output_dir / f"{x}_{y}.png"


def saveSegmentImagesAsync(segment_list, color_type, output_dir=IMAGE_OUTPUT_DIR):
    output_dir = ensureOutputDirectory(output_dir)
    write_executor = ThreadPoolExecutor()
    write_futures = [
        write_executor.submit(
            writeSegmentImage,
            segmentImagePath(segment, output_dir),
            segment.pic_matrix,
            color_type,
        )
        for segment in segment_list
    ]
    return SegmentSaveBatch(write_executor, write_futures)


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
    for x in range(num_of_pieces_width):
        for y in range(num_of_pieces_height):
            save = image[pic_x: pic_x+length, pic_y: pic_y+length, :]
            segment_to_append = Segment(save, num_of_pieces_width,
                                        num_of_pieces_height, piece_num, piece_num, score_dict, connections_dict)
            append(segment_to_append)
            piece_num += 1
            pic_y += length
        pic_x += length
        pic_y = 0
    if save_segments:
        saveSegmentImagesAsync(segments, color_type, output_dir).wait()
    return segments

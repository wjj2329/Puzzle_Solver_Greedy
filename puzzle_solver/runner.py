import random
import time

import imageio.v3 as iio
from PIL import Image
from skimage import color

from .assembly import (
    KruskalConnectionPriorityQueue,
    connectBestBudsFirst,
    findBestConnectionKruskal,
    findBestConnectionPrim,
    findBestRootSegment,
    joinPieces,
)
from .enums import AssemblyType, ColorType, CompareWithOtherSegments, ScoreAlgorithm
from .image_io import saveImage
from .paths import IMAGE_INPUT_DIR
from .scoring import calculateScores, normalizeScores
from .tiling import breakUpImage


# TODO  Multiple edge layers.  Maybe corner pixels have some extra say?
# TODO Maybe have it go in lines? Or at least start off with two lines one horizontal one vertical to build off and stop going out of bounds?
# TODO maybe combo of kruskal and prims? Divide into blocks? Limit the number of trees? Force to use prims after awhile?
# TODO do a best buddy where each piece thinks the other is the best and get those done FIRST
# TODO Different color spaces
# TODO Find balance of second best ratio
# TODO is Mahalanobis distance the same either way???? Did I get that wrong?
# TODO combo of Euclidean and Mahalanobis?
# http://chenlab.ece.cornell.edu/people/Andy/publications/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf
# https://jamesmccaffrey.wordpress.com/2017/11/09/example-of-calculating-the-mahalanobis-distance/
# https://www.python.org/dev/peps/pep-0371/ use this to make it faster
# https://www.sciencedirect.com/science/article/pii/S131915781830394X gist combo with euclidean
# https://pdfs.semanticscholar.org/4003/7d131e3365feb9d69912b3c8e8527e9ed2d5.pdf  cycle detection
# Filter the image?  Gaussian blur etc?
def main():
    start_time = time.time()
    picture_file_name = IMAGE_INPUT_DIR / "William.png"
    length = 30
    save_segments = True
    image = iio.imread(picture_file_name)
    save_assembly_to_disk = True
    show_building_animation = True
    show_print_statements = True
    boost_priority_of_big_pieces_joining = False
    connect_best_friends_first = True
    use_kruskal_priority_queue = True
    score_workers = None
    score_executor = "process"

    color_type = ColorType.LAB
    assembly_type = AssemblyType.KRUSKAL
    score_algorithm = ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS
    compare_type = CompareWithOtherSegments.ONLY_BEST
    name_for_round = "test"

    if color_type == ColorType.LAB:
        image = color.rgb2lab(image)
    segment_list = breakUpImage(
        image, length, save_segments, color_type, score_algorithm)
    calculateScores(
        segment_list, score_algorithm, show_print_statements, score_workers, score_executor)

    normalizeScores(segment_list, score_algorithm)
    elapsed_time_secs = time.time() - start_time
    if show_print_statements:
        print("Calculate scores took: %s secs " % elapsed_time_secs)
    window, w = None, None
    if show_building_animation:
        import tkinter
        from PIL import ImageTk

        window = tkinter.Tk()
        window.title("Picture")
        img = ImageTk.PhotoImage(Image.open(picture_file_name))
        w = tkinter.Label(window, image=img)
    random.shuffle(segment_list)
    round_number = 0
    original_size = len(segment_list)
    root = None
    if connect_best_friends_first:
        connectBestBudsFirst(segment_list, original_size, show_print_statements)
    if assembly_type == AssemblyType.PRIM:
        root = findBestRootSegment(segment_list)
    kruskal_queue = None
    if assembly_type == AssemblyType.KRUSKAL and use_kruskal_priority_queue:
        kruskal_queue = KruskalConnectionPriorityQueue(
            segment_list,
            boost_priority_of_big_pieces_joining,
            compare_type,
            compare_type,
        )
    while len(segment_list) > 1:
        best_connection = None
        if assembly_type == AssemblyType.KRUSKAL:
            if kruskal_queue is None:
                best_connection = findBestConnectionKruskal(
                    segment_list,
                    compare_type,
                    boost_priority_of_big_pieces_joining,
                    compare_type,
                )
            else:
                best_connection = kruskal_queue.popBestConnection(segment_list)
        if assembly_type == AssemblyType.PRIM:
            best_connection = findBestConnectionPrim(
                segment_list, root, compare_type)
        if best_connection is None or best_connection.pic_connection_matrix is None:
            break
        joinPieces(best_connection, segment_list, original_size)
        if kruskal_queue is not None:
            kruskal_queue.addConnectionsFor(best_connection.own_segment, segment_list)
        root = best_connection.own_segment
        if save_assembly_to_disk:
            image_name = saveImage(best_connection, length, round_number, color_type, name_for_round)
            if show_building_animation:
                updated_picture = ImageTk.PhotoImage(Image.open(image_name))
                w.configure(image=updated_picture)
                w.image = updated_picture
                w.pack(side="bottom", fill="both", expand="no")
                window.update()
        if show_print_statements == True:
            print("for round ", round_number, " i get score of ", best_connection.score, "the ratio for first to second best is ",
                  best_connection.score/best_connection.second_best_score, " it took ", time.time()-start_time)
        round_number += 1

    if show_print_statements == True:
        elapsed_time_secs = time.time() - start_time
        print("Execution took: %s secs " % elapsed_time_secs)

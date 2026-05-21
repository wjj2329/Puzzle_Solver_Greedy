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
from .cli import parseArguments
from .enums import AssemblyType, ColorType
from .image_io import saveImage
from .postprocess import trimAndFillAssembly
from .scoring import calculateScores, finalizeScores
from .tiling import breakUpImage


# TODO  Multiple edge layers.  Maybe corner pixels have some extra say?
# TODO Maybe have it go in lines? Or at least start off with two lines one horizontal one vertical to build off and stop going out of bounds?
# TODO maybe combo of Kruskal and Prim? Divide into blocks? Limit the number of trees? Force Prim after a while?
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
def main(argv=None):
    args = parseArguments(argv)
    start_time = time.time()
    picture_file_name = args.image
    length = args.piece_size
    image = iio.imread(picture_file_name)
    if args.color_type == ColorType.LAB:
        image = color.rgb2lab(image)
    segment_list = breakUpImage(
        image,
        length,
        args.save_segments,
        args.color_type,
        args.score_algorithm,
        output_dir=args.output_dir,
    )
    calculateScores(
        segment_list,
        args.score_algorithm,
        args.show_progress,
        args.score_workers,
        args.score_executor,
    )

    finalizeScores(segment_list, args.score_algorithm, args.score_mode)
    elapsed_time_secs = time.time() - start_time
    if args.show_progress:
        print("Score preparation took: %s secs " % elapsed_time_secs)
    window, w = None, None
    if args.show_animation:
        import tkinter
        from PIL import ImageTk

        window = tkinter.Tk()
        window.title("Picture")
        img = ImageTk.PhotoImage(Image.open(picture_file_name))
        w = tkinter.Label(window, image=img)
    if args.shuffle_seed is None:
        random.shuffle(segment_list)
    else:
        random.Random(args.shuffle_seed).shuffle(segment_list)
    round_number = 0
    original_size = len(segment_list)
    root = None
    if args.connect_best_buddy_first:
        connectBestBudsFirst(segment_list, original_size, args.show_progress)
    if args.assembly_type == AssemblyType.PRIM:
        root = findBestRootSegment(segment_list)
    kruskal_queue = None
    if (
            args.assembly_type == AssemblyType.KRUSKAL
            and args.use_kruskal_priority_queue):
        kruskal_queue = KruskalConnectionPriorityQueue(
            segment_list,
            args.boost_big_piece_priority,
            args.compare_type,
            args.compare_type,
        )
    while len(segment_list) > 1:
        best_connection = None
        if args.assembly_type == AssemblyType.KRUSKAL:
            if kruskal_queue is None:
                best_connection = findBestConnectionKruskal(
                    segment_list,
                    args.compare_type,
                    args.boost_big_piece_priority,
                    args.compare_type,
                )
            else:
                best_connection = kruskal_queue.popBestConnection(segment_list)
        if args.assembly_type == AssemblyType.PRIM:
            best_connection = findBestConnectionPrim(
                segment_list, root, args.compare_type)
        if best_connection is None or best_connection.pic_connection_matrix is None:
            break
        joinPieces(best_connection, segment_list, original_size)
        if kruskal_queue is not None:
            kruskal_queue.addConnectionsFor(
                best_connection.own_segment,
                segment_list,
            )
        root = best_connection.own_segment
        if args.save_assembly:
            image_name = saveImage(
                best_connection,
                length,
                round_number,
                args.color_type,
                args.output_name,
                output_dir=args.output_dir,
            )
            if args.show_animation:
                updated_picture = ImageTk.PhotoImage(Image.open(image_name))
                w.configure(image=updated_picture)
                w.image = updated_picture
                w.pack(side="bottom", fill="both", expand="no")
                window.update()
        if args.show_progress:
            ratio = best_connection.score / best_connection.second_best_score
            print(
                "round ",
                round_number,
                "score",
                best_connection.score,
                "first-to-second ratio",
                ratio,
                "elapsed",
                time.time() - start_time,
            )
        round_number += 1

    if args.trim_fill:
        final_connection = trimAndFillAssembly(segment_list, args.show_progress)
        if final_connection is not None and args.save_assembly:
            image_name = saveImage(
                final_connection,
                length,
                round_number,
                args.color_type,
                args.output_name,
                output_dir=args.output_dir,
            )
            if args.show_animation:
                updated_picture = ImageTk.PhotoImage(Image.open(image_name))
                w.configure(image=updated_picture)
                w.image = updated_picture
                w.pack(side="bottom", fill="both", expand="no")
                window.update()

    if args.show_progress:
        elapsed_time_secs = time.time() - start_time
        print("Execution took: %s secs " % elapsed_time_secs)

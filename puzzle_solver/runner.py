import random
import time

import imageio.v3 as iio
from PIL import Image
from skimage import color

from .assembly import (
    KruskalConnectionPriorityQueue,
    assembleKruskalBeamSearch,
    assembleKruskalHybridBeamSearch,
    connectBestBudsFirst,
    findBestConnectionKruskal,
    findBestConnectionPrim,
    findBestRootSegment,
    joinPieces,
)
from .cli import parseArguments
from .enums import AssemblyType, ColorType
from .image_io import saveImage
from .postprocess import connectEndgameComponents, trimAndFillAssembly
from .scoring import calculateScores, finalizeScores
from .tiling import breakUpImage, saveSegmentImagesAsync


def printTiming(label, started_at, show_progress):
    if show_progress:
        print(f"{label} took: {time.perf_counter() - started_at:.3f} secs")


def main(argv=None):
    args = parseArguments(argv)
    start_time = time.perf_counter()
    picture_file_name = args.image
    length = args.piece_size
    phase_started = time.perf_counter()
    image = iio.imread(picture_file_name)
    if args.color_type == ColorType.LAB:
        image = color.rgb2lab(image)
    segment_list = breakUpImage(
        image,
        length,
        False,
        args.color_type,
        output_dir=args.output_dir,
        score_storage=args.score_storage,
    )
    if args.relax_frame_bounds:
        for segment in segment_list:
            segment.enforce_frame_bounds = False
    printTiming("Image preparation", phase_started, args.show_progress)
    segment_save_batch = None
    if args.save_segments:
        phase_started = time.perf_counter()
        segment_save_batch = saveSegmentImagesAsync(
            segment_list,
            args.color_type,
            args.output_dir,
        )
        printTiming("Started segment image writes",
                    phase_started, args.show_progress)
    try:
        phase_started = time.perf_counter()
        calculateScores(
            segment_list,
            args.score_algorithm,
            args.show_progress,
            args.score_workers,
            args.score_executor,
        )
        printTiming("Score calculation", phase_started, args.show_progress)

        phase_started = time.perf_counter()
        finalizeScores(segment_list, args.score_algorithm, args.score_mode)
        printTiming("Score finalization", phase_started, args.show_progress)
    finally:
        if segment_save_batch is not None:
            phase_started = time.perf_counter()
            segment_save_batch.wait()
            printTiming("Finished segment image writes",
                        phase_started, args.show_progress)
    printTiming("Score preparation", start_time, args.show_progress)
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
        phase_started = time.perf_counter()
        connectBestBudsFirst(segment_list, original_size, args.show_progress)
        printTiming("Best-buddy setup", phase_started, args.show_progress)
    if args.assembly_type == AssemblyType.PRIM:
        root = findBestRootSegment(segment_list)
    kruskal_queue = None
    if (
            args.assembly_type == AssemblyType.KRUSKAL
            and args.beam_width <= 1
            and args.use_kruskal_priority_queue):
        phase_started = time.perf_counter()
        kruskal_queue = KruskalConnectionPriorityQueue(
            segment_list,
            args.boost_big_piece_priority,
            args.compare_type,
            args.compare_type,
        )
        printTiming("Kruskal queue build", phase_started, args.show_progress)
    assembly_started = time.perf_counter()
    if args.assembly_type == AssemblyType.KRUSKAL and args.beam_width > 1:
        def saveBeamJoin(best_connection, join_round):
            if not args.save_assembly:
                return
            image_name = saveImage(
                best_connection,
                length,
                join_round,
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

        if args.beam_start_components is None:
            round_number = assembleKruskalBeamSearch(
                segment_list,
                original_size,
                beam_width=args.beam_width,
                beam_candidates=args.beam_candidates,
                boost_priority_of_big_pieces_joining=args.boost_big_piece_priority,
                compare_type=args.compare_type,
                compare_mode=args.compare_type,
                show_progress=args.show_progress,
                on_join=saveBeamJoin if args.save_assembly else None,
            )
        else:
            round_number = assembleKruskalHybridBeamSearch(
                segment_list,
                original_size,
                beam_start_components=args.beam_start_components,
                beam_width=args.beam_width,
                beam_candidates=args.beam_candidates,
                boost_priority_of_big_pieces_joining=args.boost_big_piece_priority,
                compare_type=args.compare_type,
                compare_mode=args.compare_type,
                show_progress=args.show_progress,
                on_join=saveBeamJoin if args.save_assembly else None,
            )
        if segment_list:
            root = segment_list[0]
    while len(segment_list) > 1 and not (
            args.assembly_type == AssemblyType.KRUSKAL
            and args.beam_width > 1):
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
                time.perf_counter() - start_time,
            )
        round_number += 1
    printTiming("Assembly", assembly_started, args.show_progress)

    if args.endgame_search:
        phase_started = time.perf_counter()
        connectEndgameComponents(segment_list, show_progress=args.show_progress)
        printTiming("Endgame search", phase_started, args.show_progress)

    if args.trim_fill:
        phase_started = time.perf_counter()
        try:
            final_connection = trimAndFillAssembly(
                segment_list,
                args.show_progress,
                preserve_components=args.trim_fill_components,
                conservative_fill=args.trim_fill_conservative,
                border_tiebreak=args.trim_fill_border,
                component_frame_search=args.trim_fill_component_frame,
                edge_preserving=args.trim_fill_edge_preserving,
            )
        except KeyboardInterrupt:
            if args.show_progress:
                print("\nTrim/fill interrupted; exiting cleanly.")
            raise SystemExit(130) from None
        printTiming("Trim/fill", phase_started, args.show_progress)
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

    printTiming("Execution", start_time, args.show_progress)

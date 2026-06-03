import random
import time

import imageio.v3 as iio
import numpy as np
from PIL import Image
from skimage import color

from .assembly import (
    KruskalConnectionPriorityQueue,
    PrimConnectionPriorityQueue,
    assembleGallagherPairwiseKruskal,
    assembleGrowingConsensusKruskal,
    assembleKruskalBeamSearch,
    assembleKruskalHybridBeamSearch,
    assembleKruskalMultiContactOnly,
    assembleKruskalStaged,
    connectBestBudsFirst,
    findBestConnectionKruskal,
    findBestConnectionPrim,
    findBestPrimSeedSegment,
    joinPieces,
)
from .cli import parseArguments
from .enums import AssemblyType, ColorType, ScoreMode
from .evaluation import (
    errorDiagnosticReport,
    formatErrorDiagnosticReport,
    formatPaperStyleReport,
    paperStyleReport,
)
from .image_io import saveImage
from .postprocess import (
    connectEndgameComponents,
    placeConsensusComponentsInFrame,
    placeRepairedComponentsInFrame,
    repairConsensusShifts,
    splitBadJoinComponents,
    trimAndFillAssembly,
)
from .scoring import (
    applySymmetricCompatibilityScores,
    calculateScores,
    finalizeScores,
)
from .tiling import breakUpImage, saveSegmentImagesAsync


def printTiming(label, started_at, show_progress):
    if show_progress:
        print(f"{label} took: {time.perf_counter() - started_at:.3f} secs")


def stagedKruskalScoreLimit(args):
    if args.staged_kruskal_score_limit is not None:
        return args.staged_kruskal_score_limit
    if args.score_mode == ScoreMode.RELIABILITY:
        return 1.0
    return None


def normalizeRgbForGallagher(image):
    image = np.asarray(image, dtype=np.float64)
    if image.size > 0 and image.max() > 1.0:
        image = image / 255.0
    return image


def main(argv=None):
    args = parseArguments(argv)
    start_time = time.perf_counter()
    picture_file_name = args.image
    length = args.piece_size
    phase_started = time.perf_counter()
    image = iio.imread(picture_file_name)
    if args.color_type == ColorType.LAB:
        image = color.rgb2lab(image)
    elif args.gallagher_mode:
        image = normalizeRgbForGallagher(image)
    if args.gallagher_mode and args.show_progress:
        print(
            "Gallagher mode: RGB + MGC + reliability + Kruskal forest "
            "assembly (fixed orientation)."
        )
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
        if args.symmetric_compatibility:
            applySymmetricCompatibilityScores(segment_list)
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
        root = findBestPrimSeedSegment(
            segment_list,
            strategy=args.prim_seed_strategy,
            neighborhood_size=args.prim_seed_neighbors,
        )
        if args.show_progress and root is not None:
            print(
                "Prim seed",
                root.piece_number,
                "strategy",
                args.prim_seed_strategy,
            )
    prim_queue = None
    if (
            args.assembly_type == AssemblyType.PRIM
            and args.use_prim_priority_queue
            and root is not None):
        prim_queue = PrimConnectionPriorityQueue(
            root,
            segment_list,
            args.compare_type,
        )
    kruskal_queue = None
    if (
            args.assembly_type == AssemblyType.KRUSKAL
            and args.beam_width <= 1
            and args.use_kruskal_priority_queue
            and not args.staged_kruskal
            and not args.gallagher_pairwise_kruskal):
        phase_started = time.perf_counter()
        kruskal_queue = KruskalConnectionPriorityQueue(
            segment_list,
            args.boost_big_piece_priority,
            args.compare_type,
            args.compare_type,
        )
        printTiming("Kruskal queue build", phase_started, args.show_progress)
    assembly_started = time.perf_counter()
    if (
            args.assembly_type == AssemblyType.KRUSKAL
            and (
                args.gallagher_pairwise_kruskal
                or args.growing_consensus_kruskal
            )):
        def saveGallagherJoin(best_connection, join_round):
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

        if args.growing_consensus_kruskal:
            round_number = assembleGrowingConsensusKruskal(
                segment_list,
                original_size,
                top_candidates_per_edge=(
                    args.growing_consensus_edge_candidates
                ),
                min_support=args.growing_consensus_min_support,
                propose_missing=args.growing_consensus_propose_missing,
                max_edges=(
                    None
                    if args.growing_consensus_max_edges == 0
                    else args.growing_consensus_max_edges
                ),
                priority=args.growing_consensus_priority,
                fallback_pairwise=args.growing_consensus_fallback_pairwise,
                on_join=saveGallagherJoin if args.save_assembly else None,
            )
        else:
            round_number = assembleGallagherPairwiseKruskal(
                segment_list,
                original_size,
                top_candidates_per_edge=args.gallagher_edge_candidates,
                mutual_edges_only=args.gallagher_mutual_edges,
                on_join=saveGallagherJoin if args.save_assembly else None,
            )
        if args.repair_bad_joins:
            remerge_score_limit = args.repair_remerge_score_limit
            if remerge_score_limit is None:
                remerge_score_limit = args.repair_bad_join_score_limit
            for repair_iteration in range(args.repair_iterations):
                phase_started = time.perf_counter()
                repair_stats = splitBadJoinComponents(
                    segment_list,
                    args.repair_bad_join_score_limit,
                )
                printTiming(
                    f"Bad-join split {repair_iteration + 1}",
                    phase_started,
                    args.show_progress,
                )
                if args.show_progress:
                    print(
                        "repair split",
                        repair_iteration + 1,
                        "components",
                        repair_stats["before"],
                        "->",
                        repair_stats["after"],
                        "cut edges",
                        repair_stats["cut_edges"],
                    )
                if repair_stats["after"] <= repair_stats["before"]:
                    break

                phase_started = time.perf_counter()
                if args.repair_remerge_strategy == "multi-contact":
                    repaired_rounds = assembleKruskalMultiContactOnly(
                        segment_list,
                        original_size,
                        compare_type=args.compare_type,
                        compare_mode=args.compare_type,
                        max_score=remerge_score_limit,
                        on_join=saveGallagherJoin if args.save_assembly else None,
                    )
                else:
                    repaired_rounds = assembleGallagherPairwiseKruskal(
                        segment_list,
                        original_size,
                        top_candidates_per_edge=args.gallagher_edge_candidates,
                        mutual_edges_only=args.gallagher_mutual_edges,
                        max_score=remerge_score_limit,
                        on_join=saveGallagherJoin if args.save_assembly else None,
                    )
                round_number += repaired_rounds
                printTiming(
                    f"Bad-join remerge {repair_iteration + 1}",
                    phase_started,
                    args.show_progress,
                )
                placement_stats = None
                if args.repair_frame_placement:
                    phase_started = time.perf_counter()
                    placement_stats = placeRepairedComponentsInFrame(
                        segment_list,
                        min_neighbor_count=(
                            args.repair_frame_placement_min_contacts
                        ),
                        max_score=remerge_score_limit,
                    )
                    printTiming(
                        f"Repair frame placement {repair_iteration + 1}",
                        phase_started,
                        args.show_progress,
                    )
                    if args.show_progress:
                        print(
                            "repair frame placement",
                            repair_iteration + 1,
                            "components",
                            placement_stats["before"],
                            "->",
                            placement_stats["after"],
                            "placed",
                            placement_stats["placed_components"],
                            "components",
                            placement_stats["placed_pieces"],
                            "pieces",
                        )
                if (
                        repaired_rounds == 0
                        and (
                            placement_stats is None
                            or placement_stats["placed_components"] == 0
                        )):
                    break
        if args.repair_consensus_shifts:
            remerge_score_limit = args.repair_remerge_score_limit
            if remerge_score_limit is None and args.repair_bad_joins:
                remerge_score_limit = args.repair_bad_join_score_limit
            phase_started = time.perf_counter()
            consensus_stats = repairConsensusShifts(
                segment_list,
                top_k=args.repair_consensus_top_k,
                min_local_support=args.repair_consensus_min_local_support,
                min_neighbor_count=args.repair_consensus_min_contacts,
                max_shift=args.repair_consensus_max_shift,
                max_score=remerge_score_limit,
            )
            printTiming(
                "Consensus shift repair",
                phase_started,
                args.show_progress,
            )
            if args.show_progress:
                print(
                    "consensus shift repair",
                    "components",
                    consensus_stats["before"],
                    "->",
                    consensus_stats["after"],
                    "cut edges",
                    consensus_stats["cut_edges"],
                    "placed",
                    consensus_stats["placed_components"],
                    "components",
                    consensus_stats["placed_pieces"],
                    "pieces",
                )
            if args.repair_consensus_remerge and len(segment_list) > 1:
                phase_started = time.perf_counter()
                consensus_remerge_rounds = assembleKruskalMultiContactOnly(
                    segment_list,
                    original_size,
                    compare_type=args.compare_type,
                    compare_mode=args.compare_type,
                    max_score=remerge_score_limit,
                    on_join=saveGallagherJoin if args.save_assembly else None,
                )
                round_number += consensus_remerge_rounds
                printTiming(
                    "Consensus multi-contact remerge",
                    phase_started,
                    args.show_progress,
                )
                if args.show_progress:
                    print(
                        "consensus multi-contact remerge",
                        "rounds",
                        consensus_remerge_rounds,
                        "components now",
                        len(segment_list),
                    )
            if args.repair_consensus_frame_placement and len(segment_list) > 1:
                phase_started = time.perf_counter()
                consensus_frame_stats = placeConsensusComponentsInFrame(
                    segment_list,
                    min_neighbor_count=(
                        args.repair_consensus_frame_min_contacts
                    ),
                    max_score=remerge_score_limit,
                )
                printTiming(
                    "Consensus frame placement",
                    phase_started,
                    args.show_progress,
                )
                if args.show_progress:
                    print(
                        "consensus frame placement",
                        "components",
                        consensus_frame_stats["before"],
                        "->",
                        consensus_frame_stats["after"],
                        "placed",
                        consensus_frame_stats["placed_components"],
                        "components",
                        consensus_frame_stats["placed_pieces"],
                        "pieces",
                    )
        if segment_list:
            root = segment_list[0]
    elif args.assembly_type == AssemblyType.KRUSKAL and args.beam_width > 1:
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
    elif args.assembly_type == AssemblyType.KRUSKAL and args.staged_kruskal:
        def saveStagedJoin(best_connection, join_round):
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

        round_number = assembleKruskalStaged(
            segment_list,
            original_size,
            boost_priority_of_big_pieces_joining=args.boost_big_piece_priority,
            compare_type=args.compare_type,
            compare_mode=args.compare_type,
            max_multi_contact_score=stagedKruskalScoreLimit(args),
            on_join=saveStagedJoin if args.save_assembly else None,
        )
        if segment_list:
            root = segment_list[0]
    while len(segment_list) > 1 and not (
            args.assembly_type == AssemblyType.KRUSKAL
            and (
                args.beam_width > 1
                or args.staged_kruskal
                or args.gallagher_pairwise_kruskal
            )):
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
            if prim_queue is None:
                best_connection = findBestConnectionPrim(
                    segment_list, root, args.compare_type)
            else:
                best_connection = prim_queue.popBestConnection(
                    root,
                    segment_list,
                )
        if best_connection is None or best_connection.pic_connection_matrix is None:
            break
        joinPieces(best_connection, segment_list, original_size)
        if kruskal_queue is not None:
            kruskal_queue.addConnectionsFor(
                best_connection.own_segment,
                segment_list,
            )
        root = best_connection.own_segment
        if prim_queue is not None:
            prim_queue.addConnectionsForRoot(root, segment_list)
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
    if args.quality_report:
        report = paperStyleReport(
            segment_list,
            include_rank_stats=args.rank_report,
        )
        print(formatPaperStyleReport(report))
    if args.diagnostic_report:
        report = errorDiagnosticReport(
            segment_list,
            limit=args.diagnostic_limit,
        )
        print(formatErrorDiagnosticReport(report))

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from skimage import color

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import puzzle_solver as Solver  # noqa: E402


@dataclass(frozen=True)
class Recipe:
    name: str
    assembly_strategy: str = "queue"
    best_buddy: bool = True
    boost_big_piece_priority: bool = False
    compare_type: Solver.CompareWithOtherSegments = (
        Solver.CompareWithOtherSegments.ONLY_BEST
    )
    trim_fill: bool = True
    trim_fill_components: bool = False
    beam_width: int = 2
    beam_candidates: int | None = None
    beam_start_components: int | None = None
    score_algorithm: Solver.ScoreAlgorithm = (
        Solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS
    )
    score_mode: Solver.ScoreMode = Solver.ScoreMode.DISSIMILARITY
    color_type: Solver.ColorType = Solver.ColorType.LAB


RECIPES = {
    "baseline": Recipe("baseline"),
    "baseline_raw": Recipe("baseline_raw", trim_fill=False),
    "contact_fill": Recipe(
        "contact_fill",
        best_buddy=False,
        boost_big_piece_priority=True,
    ),
    "contact_raw": Recipe(
        "contact_raw",
        best_buddy=False,
        boost_big_piece_priority=True,
        trim_fill=False,
    ),
    "contact_compare_fill": Recipe(
        "contact_compare_fill",
        best_buddy=False,
        boost_big_piece_priority=True,
        compare_type=Solver.CompareWithOtherSegments.COMPARE_WITH_SECOND,
    ),
    "contact_compare_components": Recipe(
        "contact_compare_components",
        best_buddy=False,
        boost_big_piece_priority=True,
        compare_type=Solver.CompareWithOtherSegments.COMPARE_WITH_SECOND,
        trim_fill_components=True,
    ),
    "contact_compare_raw": Recipe(
        "contact_compare_raw",
        best_buddy=False,
        boost_big_piece_priority=True,
        compare_type=Solver.CompareWithOtherSegments.COMPARE_WITH_SECOND,
        trim_fill=False,
    ),
    "mgc_reliability": Recipe(
        "mgc_reliability",
        score_algorithm=Solver.ScoreAlgorithm.MGC,
        score_mode=Solver.ScoreMode.RELIABILITY,
    ),
    "hybrid_beam": Recipe(
        "hybrid_beam",
        assembly_strategy="beam",
        best_buddy=True,
        beam_width=2,
        beam_candidates=2,
        beam_start_components=64,
    ),
}
DEFAULT_RECIPES = (
    "baseline",
    "contact_fill",
    "contact_raw",
    "contact_compare_fill",
    "contact_compare_components",
    "contact_compare_raw",
)


SCORE_MATRIX = (
    (
        "euclidean_dissimilarity",
        Solver.ScoreAlgorithm.EUCLIDEAN,
        Solver.ScoreMode.DISSIMILARITY,
    ),
    (
        "euclidean_reliability",
        Solver.ScoreAlgorithm.EUCLIDEAN,
        Solver.ScoreMode.RELIABILITY,
    ),
    (
        "mahalanobis_dissimilarity",
        Solver.ScoreAlgorithm.MAHALANOBIS,
        Solver.ScoreMode.DISSIMILARITY,
    ),
    (
        "mahalanobis_reliability",
        Solver.ScoreAlgorithm.MAHALANOBIS,
        Solver.ScoreMode.RELIABILITY,
    ),
    (
        "combined_dissimilarity",
        Solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
        Solver.ScoreMode.DISSIMILARITY,
    ),
    (
        "combined_reliability",
        Solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
        Solver.ScoreMode.RELIABILITY,
    ),
    (
        "mgc_dissimilarity",
        Solver.ScoreAlgorithm.MGC,
        Solver.ScoreMode.DISSIMILARITY,
    ),
    (
        "mgc_reliability",
        Solver.ScoreAlgorithm.MGC,
        Solver.ScoreMode.RELIABILITY,
    ),
)

ASSEMBLY_MATRIX = (
    (
        "kruskal_bb_fill",
        {
            "assembly_strategy": "queue",
            "best_buddy": True,
            "trim_fill": True,
        },
    ),
    (
        "kruskal_bb_raw",
        {
            "assembly_strategy": "queue",
            "best_buddy": True,
            "trim_fill": False,
        },
    ),
    (
        "kruskal_no_bb_fill",
        {
            "assembly_strategy": "queue",
            "best_buddy": False,
            "trim_fill": True,
        },
    ),
    (
        "kruskal_no_bb_raw",
        {
            "assembly_strategy": "queue",
            "best_buddy": False,
            "trim_fill": False,
        },
    ),
    (
        "kruskal_contact_fill",
        {
            "assembly_strategy": "queue",
            "best_buddy": False,
            "boost_big_piece_priority": True,
            "trim_fill": True,
        },
    ),
    (
        "kruskal_contact_raw",
        {
            "assembly_strategy": "queue",
            "best_buddy": False,
            "boost_big_piece_priority": True,
            "trim_fill": False,
        },
    ),
    (
        "kruskal_contact_compare_fill",
        {
            "assembly_strategy": "queue",
            "best_buddy": False,
            "boost_big_piece_priority": True,
            "compare_type": Solver.CompareWithOtherSegments.COMPARE_WITH_SECOND,
            "trim_fill": True,
        },
    ),
    (
        "kruskal_contact_compare_raw",
        {
            "assembly_strategy": "queue",
            "best_buddy": False,
            "boost_big_piece_priority": True,
            "compare_type": Solver.CompareWithOtherSegments.COMPARE_WITH_SECOND,
            "trim_fill": False,
        },
    ),
    (
        "kruskal_contact_compare_components",
        {
            "assembly_strategy": "queue",
            "best_buddy": False,
            "boost_big_piece_priority": True,
            "compare_type": Solver.CompareWithOtherSegments.COMPARE_WITH_SECOND,
            "trim_fill": True,
            "trim_fill_components": True,
        },
    ),
    (
        "prim_bb_fill",
        {
            "assembly_strategy": "prim",
            "best_buddy": True,
            "trim_fill": True,
        },
    ),
    (
        "prim_bb_raw",
        {
            "assembly_strategy": "prim",
            "best_buddy": True,
            "trim_fill": False,
        },
    ),
    (
        "prim_no_bb_fill",
        {
            "assembly_strategy": "prim",
            "best_buddy": False,
            "trim_fill": True,
        },
    ),
    (
        "prim_no_bb_raw",
        {
            "assembly_strategy": "prim",
            "best_buddy": False,
            "trim_fill": False,
        },
    ),
)


def buildMatrixRecipes():
    recipes = []
    for assembly_name, assembly_kwargs in ASSEMBLY_MATRIX:
        for score_name, score_algorithm, score_mode in SCORE_MATRIX:
            name = f"{assembly_name}__{score_name}"
            recipes.append(Recipe(
                name,
                score_algorithm=score_algorithm,
                score_mode=score_mode,
                **assembly_kwargs,
            ))
    return recipes


MATRIX_RECIPES = buildMatrixRecipes()
MATRIX_RECIPE_NAMES = tuple(recipe.name for recipe in MATRIX_RECIPES)
RECIPES.update((recipe.name, recipe) for recipe in MATRIX_RECIPES)

CORE_SCORE_NAMES = (
    "euclidean_dissimilarity",
    "euclidean_reliability",
    "mahalanobis_dissimilarity",
    "mahalanobis_reliability",
    "combined_dissimilarity",
    "combined_reliability",
    "mgc_reliability",
)
CORE_ASSEMBLY_NAMES = (
    "kruskal_bb_fill",
    "kruskal_bb_raw",
    "kruskal_contact_fill",
    "kruskal_contact_raw",
    "kruskal_contact_compare_fill",
    "kruskal_contact_compare_raw",
    "prim_no_bb_fill",
)
CORE_MATRIX_RECIPE_NAMES = tuple(
    f"{assembly_name}__{score_name}"
    for assembly_name in CORE_ASSEMBLY_NAMES
    for score_name in CORE_SCORE_NAMES
)


def parseRecipeNames(value):
    names = []
    for item in value.split(","):
        name = item.strip()
        if name:
            names.append(name)
    unknown = [name for name in names if name not in RECIPES]
    if unknown:
        choices = ", ".join(sorted(RECIPES))
        raise argparse.ArgumentTypeError(
            f"unknown recipe(s): {', '.join(unknown)}; choose from: {choices}"
        )
    return names


def expandImageGlobs(patterns):
    images = []
    seen = set()
    for pattern in patterns:
        path = Path(pattern)
        if not path.is_absolute():
            path = ROOT / path
        matches = sorted(path.parent.glob(path.name))
        for match in matches:
            if match.is_file() and match not in seen:
                seen.add(match)
                images.append(match)
    return images


def loadImage(path, color_type):
    image = iio.imread(path)
    if image.ndim == 2:
        image = np.stack((image, image, image), axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if color_type == Solver.ColorType.LAB:
        return color.rgb2lab(image), image.shape
    return image, image.shape


def scoreKey(recipe):
    return recipe.color_type, recipe.score_algorithm, recipe.score_mode


def buildScoredSegments(image_path, piece_size, recipe, args):
    image, image_shape = loadImage(image_path, recipe.color_type)
    if image.shape[0] != image.shape[1]:
        raise ValueError("image must be square")
    if image.shape[0] % piece_size != 0:
        raise ValueError("image dimensions must be divisible by piece size")

    timings = {}
    started = time.perf_counter()
    segments = Solver.breakUpImage(
        image,
        piece_size,
        save_segments=False,
        color_type=recipe.color_type,
        score_storage=args.score_storage,
    )
    timings["break_up_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    Solver.calculateScores(
        segments,
        recipe.score_algorithm,
        show_progress=args.show_progress,
        max_workers=args.score_workers,
        executor_type=args.score_executor,
    )
    timings["score_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    Solver.finalizeScores(
        segments,
        recipe.score_algorithm,
        recipe.score_mode,
    )
    timings["finalize_seconds"] = time.perf_counter() - started
    return segments, image_shape, timings


def piecePosition(piece_number, width):
    return divmod(piece_number - 1, width)


def isCorrectNeighbor(piece, neighbor, row_delta, col_delta):
    piece_row, piece_col = piecePosition(piece.piece_number, piece.max_height)
    neighbor_row, neighbor_col = piecePosition(
        neighbor.piece_number,
        neighbor.max_height,
    )
    return (
        neighbor_row - piece_row == row_delta
        and neighbor_col - piece_col == col_delta
    )


def assemblyQuality(segment_list):
    if not segment_list:
        return {
            "adjacent": 0,
            "correct": 0,
            "incorrect": 0,
            "possible": 0,
            "coverage": 0.0,
            "precision": 0.0,
        }

    frame_height = segment_list[0].max_height
    frame_width = segment_list[0].max_width
    possible = (
        frame_height * (frame_width - 1)
        + frame_width * (frame_height - 1)
    )
    adjacent = 0
    correct = 0
    for segment in segment_list:
        matrix = segment.pic_connection_matrix
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                piece = matrix[row, col]
                if piece == 0:
                    continue
                if col + 1 < cols and matrix[row, col + 1] != 0:
                    adjacent += 1
                    if isCorrectNeighbor(piece, matrix[row, col + 1], 0, 1):
                        correct += 1
                if row + 1 < rows and matrix[row + 1, col] != 0:
                    adjacent += 1
                    if isCorrectNeighbor(piece, matrix[row + 1, col], 1, 0):
                        correct += 1
    incorrect = adjacent - correct
    return {
        "adjacent": adjacent,
        "correct": correct,
        "incorrect": incorrect,
        "possible": possible,
        "coverage": correct / possible if possible else 0.0,
        "precision": correct / adjacent if adjacent else 0.0,
    }


def assembleQueue(segments, original_size, recipe):
    queue = Solver.KruskalConnectionPriorityQueue(
        segments,
        recipe.boost_big_piece_priority,
        recipe.compare_type,
        recipe.compare_type,
    )
    rounds = 0
    while len(segments) > 1:
        best_connection = queue.popBestConnection(segments)
        if best_connection.pic_connection_matrix is None:
            break
        Solver.joinPieces(best_connection, segments, original_size)
        queue.addConnectionsFor(best_connection.own_segment, segments)
        rounds += 1
    return rounds


def assembleBeam(segments, original_size, recipe):
    if recipe.beam_start_components is None:
        return Solver.assembleKruskalBeamSearch(
            segments,
            original_size,
            beam_width=recipe.beam_width,
            beam_candidates=recipe.beam_candidates,
            boost_priority_of_big_pieces_joining=(
                recipe.boost_big_piece_priority
            ),
            compare_type=recipe.compare_type,
            compare_mode=recipe.compare_type,
            show_progress=False,
        )
    return Solver.assembleKruskalHybridBeamSearch(
        segments,
        original_size,
        beam_start_components=recipe.beam_start_components,
        beam_width=recipe.beam_width,
        beam_candidates=recipe.beam_candidates,
        boost_priority_of_big_pieces_joining=recipe.boost_big_piece_priority,
        compare_type=recipe.compare_type,
        compare_mode=recipe.compare_type,
        show_progress=False,
    )


def assemblePrim(segments, original_size, recipe):
    if not segments:
        return 0

    root = segments[0]
    rounds = 0
    while len(segments) > 1:
        best_connection = Solver.findBestConnectionPrim(
            segments,
            root,
            recipe.compare_type,
        )
        if best_connection.pic_connection_matrix is None:
            break
        Solver.joinPieces(best_connection, segments, original_size)
        root = best_connection.own_segment
        rounds += 1
    return rounds


def runRecipe(base_segments, recipe, args):
    timings = {}
    segments = Solver.cloneSegmentList(base_segments)
    random.Random(args.seed).shuffle(segments)
    original_size = len(segments)

    started = time.perf_counter()
    if recipe.best_buddy:
        Solver.connectBestBudsFirst(
            segments,
            original_size=original_size,
            show_progress=args.show_progress,
        )
    timings["best_buddy_seconds"] = time.perf_counter() - started
    after_best_buddy = len(segments)

    started = time.perf_counter()
    if recipe.assembly_strategy == "beam":
        rounds = assembleBeam(segments, original_size, recipe)
    elif recipe.assembly_strategy == "prim":
        rounds = assemblePrim(segments, original_size, recipe)
    else:
        rounds = assembleQueue(segments, original_size, recipe)
    timings["assembly_seconds"] = time.perf_counter() - started
    assembly_remaining = len(segments)

    started = time.perf_counter()
    if recipe.trim_fill:
        Solver.trimAndFillAssembly(
            segments,
            show_progress=args.show_progress,
            preserve_components=recipe.trim_fill_components,
        )
    timings["trim_fill_seconds"] = time.perf_counter() - started

    quality = assemblyQuality(segments)
    return {
        "after_best_buddy": after_best_buddy,
        "assembly_remaining": assembly_remaining,
        "remaining": len(segments),
        "rounds": rounds,
        **timings,
        **quality,
    }


def recipeRow(recipe):
    return {
        "recipe": recipe.name,
        "assembly_strategy": recipe.assembly_strategy,
        "best_buddy": recipe.best_buddy,
        "boost_big_piece_priority": recipe.boost_big_piece_priority,
        "compare_type": recipe.compare_type.name.lower(),
        "trim_fill": recipe.trim_fill,
        "trim_fill_components": recipe.trim_fill_components,
        "beam_width": recipe.beam_width,
        "beam_candidates": recipe.beam_candidates or recipe.beam_width,
        "beam_start_components": recipe.beam_start_components,
        "score_algorithm": recipe.score_algorithm.name.lower(),
        "score_mode": recipe.score_mode.name.lower(),
        "color_type": recipe.color_type.name.lower(),
    }


def benchmarkFieldnames():
    return [
        "image",
        "image_name",
        "image_shape",
        "piece_size",
        "pieces",
        "recipe",
        *[field.name for field in fields(Recipe) if field.name != "name"],
        "after_best_buddy",
        "assembly_remaining",
        "remaining",
        "rounds",
        "adjacent",
        "correct",
        "incorrect",
        "possible",
        "coverage",
        "precision",
        "break_up_seconds",
        "score_seconds",
        "finalize_seconds",
        "best_buddy_seconds",
        "assembly_seconds",
        "trim_fill_seconds",
        "total_recipe_seconds",
    ]


def runBenchmarks(args):
    image_paths = expandImageGlobs(args.image_glob)
    if args.include_william:
        william = ROOT / "input_image" / "William.png"
        if william not in image_paths:
            image_paths.insert(0, william)
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise SystemExit("No images matched the requested glob(s).")

    recipe_names = []
    for names in args.recipes:
        recipe_names.extend(names)
    recipes = [RECIPES[name] for name in recipe_names]
    output_csv = args.output_csv
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = set()
    if args.resume and output_csv.exists():
        with output_csv.open(newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                existing_keys.add((
                    row["image"],
                    int(row["piece_size"]),
                    row["recipe"],
                ))

    row_count = 0
    should_append = args.resume and output_csv.exists()
    with output_csv.open("a" if should_append else "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=benchmarkFieldnames())
        if not should_append:
            writer.writeheader()
        for piece_size in args.piece_size:
            for image_path in image_paths:
                relative_image = str(image_path.relative_to(ROOT))
                recipes_to_run = [
                    recipe for recipe in recipes
                    if (
                        relative_image,
                        piece_size,
                        recipe.name,
                    ) not in existing_keys
                ]
                if not recipes_to_run:
                    print(
                        f"Skipping {image_path.name} size={piece_size}; "
                        "all requested recipes already exist",
                        flush=True,
                    )
                    continue
                grouped_recipes = {}
                for recipe in recipes_to_run:
                    grouped_recipes.setdefault(scoreKey(recipe), []).append(
                        recipe
                    )

                for _key, score_recipes in grouped_recipes.items():
                    score_recipe = score_recipes[0]
                    print(
                        f"Scoring {image_path.name} size={piece_size} "
                        f"{score_recipe.score_algorithm.name.lower()}/"
                        f"{score_recipe.score_mode.name.lower()}",
                        flush=True,
                    )
                    scored_segments, image_shape, score_timings = (
                        buildScoredSegments(
                            image_path,
                            piece_size,
                            score_recipe,
                            args,
                        )
                    )
                    original_size = len(scored_segments)
                    for recipe in score_recipes:
                        started = time.perf_counter()
                        result = runRecipe(scored_segments, recipe, args)
                        total_seconds = time.perf_counter() - started
                        row = {
                            "image": relative_image,
                            "image_name": image_path.name,
                            "image_shape": "x".join(map(str, image_shape[:2])),
                            "piece_size": piece_size,
                            "pieces": original_size,
                            **recipeRow(recipe),
                            **score_timings,
                            **result,
                            "total_recipe_seconds": total_seconds,
                        }
                        writer.writerow(row)
                        row_count += 1
                        existing_keys.add((
                            relative_image,
                            piece_size,
                            recipe.name,
                        ))
                        csv_file.flush()
                        print(
                            f"  {recipe.name:48s} "
                            f"coverage={row['coverage']:.3f} "
                            f"precision={row['precision']:.3f} "
                            f"remaining={row['remaining']} "
                            f"seconds={total_seconds:.1f}",
                            flush=True,
                        )
    print(f"Wrote {row_count} rows to {output_csv}")


def buildParser():
    parser = argparse.ArgumentParser(
        description="Run puzzle solver recipes across a local image test set.",
    )
    parser.add_argument(
        "--image-glob",
        action="append",
        default=None,
        help=(
            "Image glob relative to repo root. Can be repeated. "
            "Default: input_image/windows_wallpaper_*.jpg"
        ),
    )
    parser.add_argument(
        "--include-william",
        action="store_true",
        help="Also include input_image/William.png before matched images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of matched images for smoke tests.",
    )
    parser.add_argument(
        "--piece-size",
        action="append",
        type=int,
        default=None,
        help="Piece size to test. Can be repeated. Default: 30.",
    )
    parser.add_argument(
        "--recipes",
        type=parseRecipeNames,
        action="append",
        default=None,
        help=(
            "Comma-separated recipe names. Can be repeated. "
            f"Default: {', '.join(DEFAULT_RECIPES)}"
        ),
    )
    parser.add_argument(
        "--recipe-set",
        choices=("default", "core-matrix", "matrix"),
        default="default",
        help=(
            "Named recipe bundle. 'core-matrix' covers the main Kruskal/Prim "
            "variants across Euclidean, Mahalanobis, combined, and MGC scores. "
            "'matrix' runs the larger exhaustive matrix."
        ),
    )
    parser.add_argument(
        "--list-recipes",
        action="store_true",
        help="Print available recipes and exit.",
    )
    parser.add_argument(
        "--score-executor",
        choices=("serial", "thread", "process"),
        default="process",
    )
    parser.add_argument("--score-workers", type=int, default=4)
    parser.add_argument(
        "--score-storage",
        choices=("dense", "dict"),
        default="dense",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output_image/test_set_benchmark.csv"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append only missing image/piece-size/recipe rows to an existing CSV.",
    )
    parser.add_argument("--show-progress", action="store_true")
    return parser


def main():
    parser = buildParser()
    args = parser.parse_args()
    if args.list_recipes:
        for name in sorted(RECIPES):
            recipe = RECIPES[name]
            print(
                f"{name}: "
                f"assembly={recipe.assembly_strategy}, "
                f"best_buddy={recipe.best_buddy}, "
                f"boost={recipe.boost_big_piece_priority}, "
                f"compare={recipe.compare_type.name.lower()}, "
                f"trim={recipe.trim_fill}, "
                f"trim_components={recipe.trim_fill_components}, "
                f"score={recipe.score_algorithm.name.lower()}/"
                f"{recipe.score_mode.name.lower()}"
            )
        return

    if args.image_glob is None:
        args.image_glob = ["input_image/windows_wallpaper_*.jpg"]
    if args.piece_size is None:
        args.piece_size = [30]
    if args.recipes is None:
        if args.recipe_set == "core-matrix":
            args.recipes = [list(CORE_MATRIX_RECIPE_NAMES)]
        elif args.recipe_set == "matrix":
            args.recipes = [list(MATRIX_RECIPE_NAMES)]
        else:
            args.recipes = [list(DEFAULT_RECIPES)]
    if any(piece_size <= 0 for piece_size in args.piece_size):
        raise SystemExit("--piece-size must be greater than 0")
    if args.score_workers is not None and args.score_workers <= 0:
        raise SystemExit("--score-workers must be greater than 0")
    runBenchmarks(args)


if __name__ == "__main__":
    main()

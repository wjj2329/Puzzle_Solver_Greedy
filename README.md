# Puzzle Solver Greedy

This project is an experimental image puzzle solver. It takes a square image, splits it into equal-sized square tiles, shuffles/handles those tiles as puzzle pieces, and then tries to reconstruct the original image by greedily matching tile edges.

It was originally written as a Python learning project, so some parts of the code are intentionally rough and exploratory. The main idea is still useful: compare possible tile adjacencies, score how well their edges match, and repeatedly join the best-looking pieces or partial assemblies.

## How It Works

At a high level, the `puzzle_solver` package does the following:

1. Loads an input image.
2. Splits the image into fixed-size square tiles.
3. Scores every possible pair of tile edges.
4. Normalizes the scores.
5. Greedily joins the best-matching pieces.
6. Saves intermediate assembly images as the puzzle is rebuilt.

The solver currently supports several scoring/assembly ideas in code:

- Euclidean color distance between neighboring edges.
- Mahalanobis distance between neighboring edges.
- A combined Euclidean plus Mahalanobis score.
- Gallagher-style Mahalanobis Gradient Compatibility scoring.
- Second-best reliability scoring.
- LAB or RGB color comparison.
- Kruskal-style greedy assembly.
- Prim-style greedy assembly.
- Trim-and-fill cleanup for holes left by the greedy assembly.

The default path uses LAB color, combined Euclidean/Mahalanobis scoring, Kruskal-style assembly, and trim-and-fill cleanup.

## Repository Layout

```text
.
├── puzzle_solver/
│   ├── __main__.py        # Module entry point for python3 -m puzzle_solver
│   ├── assembly.py        # Kruskal/Prim/best-buddy assembly logic
│   ├── cli.py             # Command-line argument setup
│   ├── distances.py       # Euclidean and Mahalanobis distance helpers
│   ├── enums.py           # Algorithm, color, direction, and assembly enums
│   ├── image_io.py        # Output directory and image writing helpers
│   ├── models.py          # Segment, BestConnection, ScoreEdge, ScorePayload
│   ├── paths.py           # Project input/output paths
│   ├── runner.py          # Main solver workflow
│   ├── score_helpers.py   # Pairwise edge score helpers
│   ├── scoring.py         # Serial/thread/process score calculation
│   ├── solver.py          # Direct-file entry point
│   └── tiling.py          # Image splitting helpers
├── input_image/
│   └── William.png    # Example input image
├── output_image/     # Generated puzzle pieces and assembly snapshots
├── tests/
│   └── test_solver_baseline.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Requirements

This code depends on several scientific/image-processing Python packages:

- `numpy`
- `scipy`
- `imageio`
- `Pillow`
- `scikit-image`

`tkinter` is also used for the optional build animation. It is included with many Python installations, but some environments require installing it separately.

Because this is older exploratory code, it may need small dependency/version fixes on modern Python versions.

Install the Python dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Running

From the repository root:

```bash
python3 -m puzzle_solver
```

You can also run the direct-file entry point:

```bash
python3 puzzle_solver/solver.py
```

Show all solver options with:

```bash
python3 -m puzzle_solver -h
```

The no-argument solver run is equivalent to:

```bash
python3 -m puzzle_solver \
  --image input_image/William.png \
  --piece-size 30 \
  --save-segments \
  --save-assembly \
  --animation \
  --progress \
  --best-buddy \
  --kruskal-priority-queue \
  --trim-fill \
  --score-executor process \
  --score-storage dense \
  --color-type lab \
  --assembly-type kruskal \
  --score-algorithm euclidean_and_mahalanobis \
  --score-mode dissimilarity \
  --compare-type only_best \
  --output-name test
```

For a faster non-GUI run while experimenting:

```bash
python3 -m puzzle_solver --piece-size 120 --score-mode reliability --no-animation --no-save-assembly --no-save-segments --no-progress
```

`--score-workers` omitted uses the available CPU count for score calculation.
Set it to `1` to force serial scoring. `--score-executor` can be `"serial"`,
`"thread"`, or `"process"`. The default `"process"` backend gives true
multi-core parallelism for the score-calculation phase.

`--score-storage dense` stores pairwise scores in a dense NumPy-backed table
while preserving the same `(piece, direction, piece)` lookups used by assembly.
Use `--score-storage dict` to run with the original dictionary storage.

`--kruskal-priority-queue` uses the faster priority-queue assembly path. Use
`--no-kruskal-priority-queue` for the older full-scan Kruskal loop.

`--score-mode dissimilarity` uses the original lower-is-better edge costs.
Use `--score-mode reliability` to convert those costs into second-best
reliability scores before best-buddy and assembly.

`--score-algorithm mgc` uses Gallagher-style Mahalanobis Gradient
Compatibility. It compares the seam gradient against each piece's internal
edge-gradient distribution, then sums both directions before optional
second-best reliability scoring. A paper-style run is:

```bash
python3 -m puzzle_solver --score-algorithm mgc --score-mode reliability
```

`--trim-fill` runs a Gallagher-style cleanup after greedy assembly: the largest
assembled tree is trimmed to the known puzzle frame and remaining holes are
filled from leftover or trimmed pieces by neighbor compatibility. Use
`--no-trim-fill` to inspect the raw greedy tree output.

Generated tile and assembly images are written to the repo-level `output_image/` directory.

To benchmark score-calculation backends on generated image data:

```bash
python3 benchmarks/benchmark_scores.py --image-size 960 --piece-size 30 --runs serial:1 process:4 process:8
```

To benchmark the best-buddy pre-assembly pass:

```bash
python3 benchmarks/benchmark_best_buddy.py --image-size 240 --piece-size 30 --repeat 3
```

To benchmark the round-by-round Kruskal assembly loop after score calculation
and best-buddy setup:

```bash
python3 benchmarks/benchmark_kruskal_assembly.py --image-size 240 --piece-size 30
```

That benchmark uses a smooth synthetic gradient image by default. Use
`--image-mode random` if you want to stress-test the solver on noisier input.

To profile a full solver run by phase:

```bash
python3 benchmarks/profile_solver_run.py --piece-size 120 --score-workers 4
```

Use `--piece-size 30` to profile the current full-size runner default; it can
take several minutes. Add `--save-segments` or `--save-assembly` when you want
to measure image-output overhead too.

The benchmark and profiling scripts default to `--score-mode dissimilarity`.
Add `--score-mode reliability` to compare the new second-best reliability mode.

## Testing

After installing the dependencies, run the baseline test suite from the repository root:

```bash
python3 -m unittest discover
```

## Input Image Notes

The current implementation expects:

- A square image.
- Image dimensions evenly divisible by the tile size.
- Square tiles.

For example, the bundled `William.png` is `960x960`, so a tile length of `30` creates a `32x32` tile puzzle.

## Generated Files

Depending on the settings in `main()`, running the solver can generate:

- Individual tile images such as `output_image/0_0.png`, `output_image/0_1.png`, etc.
- Assembly snapshots such as `output_image/test round0.png`, `output_image/test round1.png`, etc.

These generated artifacts are ignored by Git.

## Current Limitations

- Some scoring strategies are experimental and need more benchmark coverage.
- The solver is computationally expensive for larger tile counts.
- `Segment` still owns a lot of behavior and could be simplified further.

## Possible Cleanup Ideas

- Add a `pyproject.toml`.
- Shrink `Segment` into a smaller data object and move the remaining behavior into services/functions.
- Expand tests around full puzzle assembly, scoring comparisons, and known edge cases.

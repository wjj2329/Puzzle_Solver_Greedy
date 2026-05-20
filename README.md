# Puzzle Solver Greedy

This project is an experimental image puzzle solver. It takes a square image, splits it into equal-sized square tiles, shuffles/handles those tiles as puzzle pieces, and then tries to reconstruct the original image by greedily matching tile edges.

It was originally written as a Python learning project, so some parts of the code are intentionally rough and exploratory. The main idea is still useful: compare possible tile adjacencies, score how well their edges match, and repeatedly join the best-looking pieces or partial assemblies.

## How It Works

At a high level, `puzzle_solver/Solver.py` does the following:

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
- LAB or RGB color comparison.
- Kruskal-style greedy assembly.
- Prim-style greedy assembly.
- Early exploratory GIST-based scoring code.

The default path in `main()` uses LAB color, combined Euclidean/Mahalanobis scoring, and Kruskal-style assembly.

## Repository Layout

```text
.
├── puzzle_solver/
│   ├── Solver.py      # Main puzzle splitting, scoring, and assembly code
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
pip install -r requirements.txt
```

## Running

From the repository root:

```bash
python puzzle_solver/Solver.py
```

The solver is currently configured by editing variables inside `main()` in `puzzle_solver/Solver.py`.

Important defaults:

```python
picture_file_name = IMAGE_INPUT_DIR / "William.png"
length = 30
save_segments = True
save_assembly_to_disk = True
show_building_animation = True
show_print_statements = True
score_workers = None
score_executor = "process"
color_type = ColorType.LAB
assembly_type = AssemblyType.KRUSKAL
score_algorithm = ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS
name_for_round = "test"
```

`score_workers = None` uses the available CPU count for score calculation. Set it to `1` to force serial scoring. `score_executor` can be `"serial"`, `"thread"`, or `"process"`. The default `"process"` backend gives true multi-core parallelism for the score-calculation phase.

Generated tile and assembly images are written to the repo-level `output_image/` directory.

To benchmark score-calculation backends on generated image data:

```bash
python benchmarks/benchmark_scores.py --image-size 960 --piece-size 30 --runs serial:1 process:4 process:8
```

## Testing

After installing the dependencies, run the baseline test suite from the repository root:

```bash
python -m unittest discover
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
- GIST output files if the GIST path is used.

These generated artifacts are ignored by Git.

## Current Limitations

- Command-line argument parsing exists but is not wired into `main()`.
- Some scoring strategies are experimental or marked as unfinished.
- The GIST workflow references a Windows-specific executable/path.
- The solver is computationally expensive for larger tile counts.
- The code has not yet been refactored into smaller modules.

## Possible Cleanup Ideas

- Wire `argparse` into `main()` so the image, tile size, and algorithm can be selected from the command line.
- Add a `pyproject.toml`.
- Separate image splitting, scoring, assembly, and rendering into modules.
- Add a deterministic shuffle seed for repeatable runs.
- Expand tests around full puzzle assembly and known edge cases.

import contextlib
import io
import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Install requirements.txt to run solver tests") from exc


ROOT = Path(__file__).resolve().parents[1]


def load_solver():
    sys.path.insert(0, str(ROOT))
    try:
        import puzzle_solver
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest("Install requirements.txt to run solver tests") from exc
    return puzzle_solver


solver = load_solver()


class CliTests(unittest.TestCase):
    def test_solver_cli_defaults_match_runner_defaults(self):
        args = solver.parseArguments([])

        self.assertEqual(solver.IMAGE_INPUT_DIR / "William.png", args.image)
        self.assertEqual(30, args.piece_size)
        self.assertTrue(args.save_segments)
        self.assertTrue(args.save_assembly)
        self.assertTrue(args.show_animation)
        self.assertTrue(args.show_progress)
        self.assertTrue(args.connect_best_buddy_first)
        self.assertTrue(args.use_kruskal_priority_queue)
        self.assertTrue(args.trim_fill)
        self.assertEqual("process", args.score_executor)
        self.assertIsNone(args.score_workers)
        self.assertEqual(solver.ColorType.LAB, args.color_type)
        self.assertEqual(solver.AssemblyType.KRUSKAL, args.assembly_type)
        self.assertEqual(
            solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
            args.score_algorithm,
        )
        self.assertEqual(solver.ScoreMode.DISSIMILARITY, args.score_mode)
        self.assertEqual(
            solver.CompareWithOtherSegments.ONLY_BEST,
            args.compare_type,
        )

    def test_solver_cli_parses_runtime_options(self):
        args = solver.parseArguments([
            "--image",
            "input_image/smooth_gradient.png",
            "--piece-size",
            "120",
            "--no-save-segments",
            "--no-save-assembly",
            "--no-animation",
            "--no-progress",
            "--no-best-buddy",
            "--no-kruskal-priority-queue",
            "--no-trim-fill",
            "--boost-big-piece-priority",
            "--score-workers",
            "4",
            "--score-executor",
            "thread",
            "--color-type",
            "rgb",
            "--assembly-type",
            "prim",
            "--score-algorithm",
            "euclidean-and-mahalanobis",
            "--score-mode",
            "reliability",
            "--compare-type",
            "compare-with-second",
            "--shuffle-seed",
            "123",
        ])

        self.assertEqual(Path("input_image/smooth_gradient.png"), args.image)
        self.assertEqual(120, args.piece_size)
        self.assertFalse(args.save_segments)
        self.assertFalse(args.save_assembly)
        self.assertFalse(args.show_animation)
        self.assertFalse(args.show_progress)
        self.assertFalse(args.connect_best_buddy_first)
        self.assertFalse(args.use_kruskal_priority_queue)
        self.assertFalse(args.trim_fill)
        self.assertTrue(args.boost_big_piece_priority)
        self.assertEqual(4, args.score_workers)
        self.assertEqual("thread", args.score_executor)
        self.assertEqual(solver.ColorType.RGB, args.color_type)
        self.assertEqual(solver.AssemblyType.PRIM, args.assembly_type)
        self.assertEqual(
            solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
            args.score_algorithm,
        )
        self.assertEqual(solver.ScoreMode.RELIABILITY, args.score_mode)
        self.assertEqual(
            solver.CompareWithOtherSegments.COMPARE_WITH_SECOND,
            args.compare_type,
        )
        self.assertEqual(123, args.shuffle_seed)

    def test_solver_cli_help_describes_new_score_mode(self):
        help_text = solver.buildArgumentParser().format_help()

        self.assertIn("--score-mode", help_text)
        self.assertIn("second-best reliability", help_text)
        self.assertIn("--no-animation", help_text)


def legacy_mahalanobis_distance(a, a2, z, z2):
    cov = np.linalg.pinv(np.cov(a.T))
    cov2 = np.linalg.pinv(np.cov(z.T))

    red_average_1 = np.average(a[:, 0] - a2[:, 0])
    green_average_1 = np.average(a[:, 1] - a2[:, 1])
    blue_average_1 = np.average(a[:, 2] - a2[:, 2])

    red_average_2 = np.average(z[:, 0] - z2[:, 0])
    green_average_2 = np.average(z[:, 1] - z2[:, 1])
    blue_average_2 = np.average(z[:, 2] - z2[:, 2])

    score = 0.0
    testr1 = a[:, 0] - z[:, 0]
    testg1 = a[:, 1] - z[:, 1]
    testb1 = a[:, 2] - z[:, 2]

    testr2 = z[:, 0] - a[:, 0]
    testg2 = z[:, 1] - a[:, 1]
    testb2 = z[:, 2] - a[:, 2]

    for i in range(len(a)):
        mymatrix = np.asarray(
            [
                testr1[i] - red_average_1,
                testg1[i] - green_average_1,
                testb1[i] - blue_average_1,
            ]
        )
        mymatrix2 = np.asarray(
            [
                testr2[i] - red_average_2,
                testg2[i] - green_average_2,
                testb2[i] - blue_average_2,
            ]
        )
        score += math.sqrt(abs(float(mymatrix @ cov2 @ mymatrix.T)))
        score += math.sqrt(abs(float(mymatrix2 @ cov @ mymatrix2.T)))
    return score


class BreakUpImageTests(unittest.TestCase):
    def test_breaks_square_image_into_numbered_tiles(self):
        image = np.arange(4 * 4 * 3).reshape((4, 4, 3))

        segments = solver.breakUpImage(
            image,
            length=2,
            save_segments=False,
            color_type=solver.ColorType.RGB,
            score_algorithm=solver.ScoreAlgorithm.EUCLIDEAN,
        )

        self.assertEqual(4, len(segments))
        self.assertEqual([1, 2, 3, 4], [segment.piece_number for segment in segments])
        np.testing.assert_array_equal(image[0:2, 0:2, :], segments[0].pic_matrix)
        np.testing.assert_array_equal(image[0:2, 2:4, :], segments[1].pic_matrix)
        np.testing.assert_array_equal(image[2:4, 0:2, :], segments[2].pic_matrix)
        np.testing.assert_array_equal(image[2:4, 2:4, :], segments[3].pic_matrix)

    def test_rejects_non_square_images(self):
        image = np.zeros((2, 4, 3))

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                solver.breakUpImage(
                    image,
                    length=2,
                    save_segments=False,
                    color_type=solver.ColorType.RGB,
                    score_algorithm=solver.ScoreAlgorithm.EUCLIDEAN,
                )

    def test_rejects_images_not_evenly_divisible_by_tile_size(self):
        image = np.zeros((5, 5, 3))

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                solver.breakUpImage(
                    image,
                    length=2,
                    save_segments=False,
                    color_type=solver.ColorType.RGB,
                    score_algorithm=solver.ScoreAlgorithm.EUCLIDEAN,
                )


class ScoreTests(unittest.TestCase):
    def make_segment(self, matrix, piece_number, score_dict=None):
        if score_dict is None:
            score_dict = {}
        return solver.Segment(
            np.asarray(matrix),
            max_width=2,
            max_height=2,
            piece_number=piece_number,
            component_id=piece_number,
            score_dict=score_dict,
            gist=None,
            connections_dict={},
        )

    def test_euclidean_scores_are_recorded_for_both_directions(self):
        score_dict = {}
        first = self.make_segment(
            [
                [[1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4]],
            ],
            piece_number=1,
            score_dict=score_dict,
        )
        second = self.make_segment(
            [
                [[4, 4, 4], [3, 3, 3]],
                [[2, 2, 2], [1, 1, 1]],
            ],
            piece_number=2,
            score_dict=score_dict,
        )

        first.calculateScoreEuclidean(second)

        right_score = score_dict[1, solver.JoinDirection.RIGHT, 2]
        reciprocal_left_score = score_dict[2, solver.JoinDirection.LEFT, 1]
        expected_right_score = first.euclideanDistance(
            first.pic_matrix[:, -1, :].astype(np.int16),
            second.pic_matrix[:, 0, :],
        )
        self.assertEqual(expected_right_score, right_score)
        self.assertEqual(right_score, reciprocal_left_score)

        down_score = score_dict[1, solver.JoinDirection.DOWN, 2]
        reciprocal_up_score = score_dict[2, solver.JoinDirection.UP, 1]
        self.assertEqual(down_score, reciprocal_up_score)

    def test_euclidean_distance_matches_legacy_pixel_sum(self):
        first = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 6.0, 8.0],
                [2.0, 3.0, 4.0],
            ]
        )
        second = np.asarray(
            [
                [2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0],
                [5.0, 7.0, 9.0],
            ]
        )
        segment = self.make_segment(np.zeros((3, 3, 3)), piece_number=1)
        expected = sum(np.linalg.norm(x - y) for x, y in zip(first, second))

        self.assertEqual(expected, segment.euclideanDistance(first, second))

    def test_euclidean_distance_matches_legacy_first_axis_sum_for_3d_inputs(self):
        first = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        second = first + 2
        segment = self.make_segment(np.zeros((3, 3, 3)), piece_number=1)
        expected = sum(np.linalg.norm(x - y) for x, y in zip(first, second))

        self.assertEqual(expected, segment.euclideanDistance(first, second))

    def test_euclidean_distance_prevents_unsigned_byte_underflow(self):
        first = np.asarray([[0, 10, 20]], dtype=np.uint8)
        second = np.asarray([[255, 5, 25]], dtype=np.uint8)

        self.assertEqual(
            np.linalg.norm(np.asarray([-255.0, 5.0, -5.0])),
            solver.euclideanDistance(first, second),
        )

    def test_score_edges_preserve_float_precision(self):
        segment = self.make_segment(
            np.asarray(
                [
                    [[1.2, 2.3, 3.4], [4.5, 5.6, 6.7]],
                    [[7.8, 8.9, 9.1], [10.2, 11.3, 12.4]],
                ]
            ),
            piece_number=1,
        )

        right_edge = segment.ownScoreEdges()[solver.JoinDirection.RIGHT]

        self.assertTrue(np.issubdtype(right_edge.edge.dtype, np.floating))
        np.testing.assert_allclose(right_edge.edge, segment.pic_matrix[:, -1, :])

    def test_score_edges_prevent_unsigned_byte_underflow(self):
        edge = np.asarray(
            [
                [0, 10, 20],
                [5, 15, 25],
                [10, 20, 30],
            ],
            dtype=np.uint8,
        )
        adjacent_edge = np.asarray(
            [
                [255, 5, 25],
                [250, 10, 30],
                [245, 15, 35],
            ],
            dtype=np.uint8,
        )

        score_edge = solver.ScoreEdge(edge, adjacent_edge)

        np.testing.assert_allclose(
            np.asarray([-245.0, 5.0, -5.0]),
            score_edge.average_delta,
        )

    def test_mahalanobis_distance_matches_legacy_pixel_loop(self):
        segment = self.make_segment(np.zeros((4, 4, 3)), piece_number=1)
        a = np.asarray(
            [
                [0.0, 1.0, 2.0],
                [2.0, 3.0, 5.0],
                [4.0, 8.0, 13.0],
                [7.0, 11.0, 17.0],
            ]
        )
        a2 = a + np.asarray([1.0, -1.0, 2.0])
        z = np.asarray(
            [
                [3.0, 5.0, 7.0],
                [6.0, 10.0, 15.0],
                [9.0, 14.0, 23.0],
                [12.0, 20.0, 31.0],
            ]
        )
        z2 = z + np.asarray([-2.0, 1.0, -1.0])

        expected = legacy_mahalanobis_distance(a, a2, z, z2)

        self.assertAlmostEqual(expected, segment.mahalanobisDistance(a, a2, z, z2))

    def test_mahalanobis_scores_are_recorded_for_both_directions(self):
        score_dict = {}
        first = self.make_segment(
            np.arange(27).reshape((3, 3, 3)),
            piece_number=1,
            score_dict=score_dict,
        )
        second = self.make_segment(
            np.arange(27, 54).reshape((3, 3, 3)),
            piece_number=2,
            score_dict=score_dict,
        )

        first.calculateScoreMahalanobis(second)

        right_score = score_dict[1, solver.JoinDirection.RIGHT, 2]
        reciprocal_left_score = score_dict[2, solver.JoinDirection.LEFT, 1]
        expected_right_score = first.mahalanobisDistance(
            first.pic_matrix.astype(np.int16)[:, -1, :],
            first.pic_matrix.astype(np.int16)[:, -2, :],
            second.pic_matrix[:, 0, :],
            second.pic_matrix[:, 1, :],
        )
        self.assertTrue(np.isfinite(right_score))
        self.assertEqual(expected_right_score, right_score)
        self.assertEqual(right_score, reciprocal_left_score)

    def test_combined_scores_match_direct_distance_helpers(self):
        score_dict = {}
        first = self.make_segment(
            np.arange(27).reshape((3, 3, 3)),
            piece_number=1,
            score_dict=score_dict,
        )
        second = self.make_segment(
            np.arange(27, 54).reshape((3, 3, 3)),
            piece_number=2,
            score_dict=score_dict,
        )

        first.calculateScoreEuclideanAndMahalanobis(second)

        expected_right_score = (
            first.mahalanobisDistance(
                first.pic_matrix.astype(np.int16)[:, -1, :],
                first.pic_matrix.astype(np.int16)[:, -2, :],
                second.pic_matrix[:, 0, :],
                second.pic_matrix[:, 1, :],
            ),
            first.euclideanDistance(
                first.pic_matrix.astype(np.int16)[:, -1, :],
                second.pic_matrix[:, 0, :],
            ),
        )
        self.assertEqual(
            expected_right_score,
            score_dict[1, solver.JoinDirection.RIGHT, 2],
        )

    def test_combined_lab_scores_do_not_truncate_float_edges(self):
        score_dict = {}
        first = self.make_segment(
            np.asarray(
                [
                    [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]],
                    [[2.1, 3.2, 4.3], [5.4, 6.5, 7.6], [8.7, 9.8, 10.9]],
                    [[3.1, 4.2, 5.3], [6.4, 7.5, 8.6], [9.7, 10.8, 11.9]],
                ]
            ),
            piece_number=1,
            score_dict=score_dict,
        )
        second = self.make_segment(
            np.asarray(
                [
                    [[1.6, 2.7, 3.8], [4.9, 6.0, 7.1], [8.2, 9.3, 10.4]],
                    [[2.6, 3.7, 4.8], [5.9, 7.0, 8.1], [9.2, 10.3, 11.4]],
                    [[3.6, 4.7, 5.8], [6.9, 8.0, 9.1], [10.2, 11.3, 12.4]],
                ]
            ),
            piece_number=2,
            score_dict=score_dict,
        )

        first.calculateScoreEuclideanAndMahalanobis(second)

        expected_right_score = (
            first.mahalanobisDistance(
                first.pic_matrix[:, -1, :],
                first.pic_matrix[:, -2, :],
                second.pic_matrix[:, 0, :],
                second.pic_matrix[:, 1, :],
            ),
            first.euclideanDistance(
                first.pic_matrix[:, -1, :],
                second.pic_matrix[:, 0, :],
            ),
        )
        truncated_euclidean_score = first.euclideanDistance(
            first.pic_matrix.astype(np.int16)[:, -1, :],
            second.pic_matrix[:, 0, :],
        )

        self.assertEqual(
            expected_right_score,
            score_dict[1, solver.JoinDirection.RIGHT, 2],
        )
        self.assertNotEqual(
            truncated_euclidean_score,
            score_dict[1, solver.JoinDirection.RIGHT, 2][1],
        )

    def test_calculate_scores_can_run_without_progress_output(self):
        score_dict = {}
        first = self.make_segment(
            np.arange(27).reshape((3, 3, 3)),
            piece_number=1,
            score_dict=score_dict,
        )
        second = self.make_segment(
            np.arange(27, 54).reshape((3, 3, 3)),
            piece_number=2,
            score_dict=score_dict,
        )

        with contextlib.redirect_stdout(io.StringIO()) as output:
            solver.calculateScores(
                [first, second],
                solver.ScoreAlgorithm.EUCLIDEAN,
                show_progress=False,
            )

        self.assertEqual("", output.getvalue())
        self.assertIn((1, solver.JoinDirection.RIGHT, 2), score_dict)

    def test_threaded_calculate_scores_matches_serial_results(self):
        def make_segments(score_dict):
            return [
                self.make_segment(
                    np.arange(27).reshape((3, 3, 3)) + offset,
                    piece_number=index,
                    score_dict=score_dict,
                )
                for index, offset in enumerate([0, 7, 19, 31], start=1)
            ]

        serial_score_dict = {}
        threaded_score_dict = {}
        serial_segments = make_segments(serial_score_dict)
        threaded_segments = make_segments(threaded_score_dict)

        solver.calculateScores(
            serial_segments,
            solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
            show_progress=False,
            max_workers=1,
            executor_type="serial",
        )
        solver.calculateScores(
            threaded_segments,
            solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
            show_progress=False,
            max_workers=2,
            executor_type="thread",
        )

        self.assertEqual(serial_score_dict, threaded_score_dict)

    def test_process_calculate_scores_matches_serial_results(self):
        def make_segments(score_dict):
            return [
                self.make_segment(
                    np.arange(27).reshape((3, 3, 3)) + offset,
                    piece_number=index,
                    score_dict=score_dict,
                )
                for index, offset in enumerate([0, 7, 19, 31], start=1)
            ]

        serial_score_dict = {}
        process_score_dict = {}
        serial_segments = make_segments(serial_score_dict)
        process_segments = make_segments(process_score_dict)

        solver.calculateScores(
            serial_segments,
            solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
            show_progress=False,
            max_workers=1,
            executor_type="serial",
        )
        try:
            solver.calculateScores(
                process_segments,
                solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
                show_progress=False,
                max_workers=2,
                executor_type="process",
            )
        except PermissionError as exc:
            raise unittest.SkipTest("ProcessPoolExecutor is unavailable") from exc

        self.assertEqual(serial_score_dict, process_score_dict)

    def test_combined_score_normalization_preserves_relative_baseline(self):
        segment = self.make_segment(np.zeros((2, 2, 3)), piece_number=1)
        segment.score_dict.update(
            {
                ("low",): (1.0, 10.0),
                ("mid",): (2.0, 20.0),
                ("high",): (3.0, 30.0),
            }
        )

        solver.normalizeScores(
            [segment],
            solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
        )

        self.assertEqual(0.0, segment.score_dict[("low",)])
        self.assertEqual(1.0, segment.score_dict[("mid",)])
        self.assertEqual(2.0, segment.score_dict[("high",)])

    def test_reliability_score_mode_uses_second_best_cost_by_direction(self):
        segment = self.make_segment(np.zeros((2, 2, 3)), piece_number=1)
        segment.score_dict.update(
            {
                (1, solver.JoinDirection.RIGHT, 2): 2.0,
                (1, solver.JoinDirection.RIGHT, 3): 10.0,
                (2, solver.JoinDirection.LEFT, 1): 2.0,
                (2, solver.JoinDirection.LEFT, 3): 4.0,
            }
        )

        solver.applyScoreMode([segment], solver.ScoreMode.RELIABILITY)

        self.assertEqual(
            0.2,
            segment.score_dict[1, solver.JoinDirection.RIGHT, 2],
        )
        self.assertEqual(
            1.0,
            segment.score_dict[1, solver.JoinDirection.RIGHT, 3],
        )
        self.assertEqual(
            0.5,
            segment.score_dict[2, solver.JoinDirection.LEFT, 1],
        )
        self.assertEqual(
            1.0,
            segment.score_dict[2, solver.JoinDirection.LEFT, 3],
        )

    def test_reliability_score_mode_treats_single_candidate_as_ambiguous(self):
        segment = self.make_segment(np.zeros((2, 2, 3)), piece_number=1)
        segment.score_dict[1, solver.JoinDirection.UP, 2] = 7.0

        solver.applyScoreMode([segment], solver.ScoreMode.RELIABILITY)

        self.assertEqual(1.0, segment.score_dict[1, solver.JoinDirection.UP, 2])

    def test_finalize_scores_can_normalize_then_apply_reliability(self):
        segment = self.make_segment(np.zeros((2, 2, 3)), piece_number=1)
        segment.score_dict.update(
            {
                (1, solver.JoinDirection.RIGHT, 2): (1.0, 10.0),
                (1, solver.JoinDirection.RIGHT, 3): (2.0, 20.0),
                (1, solver.JoinDirection.RIGHT, 4): (3.0, 30.0),
            }
        )

        solver.finalizeScores(
            [segment],
            solver.ScoreAlgorithm.EUCLIDEAN_AND_MAHALANOBIS,
            solver.ScoreMode.RELIABILITY,
        )

        self.assertEqual(
            0.0,
            segment.score_dict[1, solver.JoinDirection.RIGHT, 2],
        )
        self.assertEqual(
            1.0,
            segment.score_dict[1, solver.JoinDirection.RIGHT, 3],
        )
        self.assertEqual(
            2.0,
            segment.score_dict[1, solver.JoinDirection.RIGHT, 4],
        )


class ImageWriteTests(unittest.TestCase):
    def test_prepare_image_for_write_converts_unit_float_images_to_uint8(self):
        image = np.asarray([[[0.0, 0.5, 1.0]]])

        prepared = solver.prepareImageForWrite(image)

        self.assertEqual(np.uint8, prepared.dtype)
        np.testing.assert_array_equal(np.asarray([[[0, 128, 255]]], dtype=np.uint8), prepared)

    def test_prepare_image_for_write_converts_float_rgb_images_to_uint8(self):
        image = np.asarray([[[-10.0, 127.4, 300.0]]])

        prepared = solver.prepareImageForWrite(image)

        self.assertEqual(np.uint8, prepared.dtype)
        np.testing.assert_array_equal(np.asarray([[[0, 127, 255]]], dtype=np.uint8), prepared)

    def test_break_up_image_writes_saved_segments_to_output_directory(self):
        image = np.arange(2 * 2 * 3, dtype=np.uint8).reshape((2, 2, 3))

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output_image"

            solver.breakUpImage(
                image,
                length=1,
                save_segments=True,
                color_type=solver.ColorType.RGB,
                score_algorithm=solver.ScoreAlgorithm.EUCLIDEAN,
                output_dir=output_dir,
            )

            self.assertEqual(
                ["0_0.png", "0_1.png", "1_0.png", "1_1.png"],
                sorted(path.name for path in output_dir.glob("*.png")),
            )

    def test_save_image_writes_assembly_snapshot_to_output_directory(self):
        segment = solver.Segment(
            np.full((2, 2, 3), 10, dtype=np.uint8),
            max_width=1,
            max_height=1,
            piece_number=1,
            component_id=1,
            score_dict={},
            gist=None,
            connections_dict={},
        )
        connection = solver.BestConnection(
            pic_connection_matrix=np.asarray([[segment]], dtype=object),
            binary_connection_matrix=np.asarray([[1]]),
        )

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output_image"

            image_path = Path(solver.saveImage(
                connection,
                piece_size=2,
                round_number=0,
                color_type=solver.ColorType.RGB,
                name_for_round="test",
                output_dir=output_dir,
            ))

            self.assertEqual(output_dir, image_path.parent)
            self.assertEqual("test round0.png", image_path.name)
            self.assertTrue(image_path.exists())


class PostProcessTests(unittest.TestCase):
    def make_segment(self, piece_number, score_dict, max_size=2):
        return solver.Segment(
            np.full((2, 2, 3), piece_number, dtype=np.uint8),
            max_width=max_size,
            max_height=max_size,
            piece_number=piece_number,
            component_id=piece_number,
            score_dict=score_dict,
            gist=None,
            connections_dict={},
        )

    def test_trim_to_best_frame_keeps_densest_known_puzzle_window(self):
        score_dict = {}
        first = self.make_segment(1, score_dict)
        second = self.make_segment(2, score_dict)
        third = self.make_segment(3, score_dict)
        root = self.make_segment(4, score_dict)
        root.pic_connection_matrix = np.asarray(
            [
                [first, second, third],
                [0, 0, 0],
            ],
            dtype=object,
        )
        root.binary_connection_matrix = (root.pic_connection_matrix != 0).astype(int)

        trimmed = solver.trimToBestFrame(root)

        self.assertEqual([3], [piece.piece_number for piece in trimmed])
        self.assertEqual((2, 2), root.pic_connection_matrix.shape)
        self.assertIs(root.pic_connection_matrix[0, 0], first)
        self.assertIs(root.pic_connection_matrix[0, 1], second)

    def test_fill_holes_uses_best_candidate_against_occupied_neighbors(self):
        score_dict = {}
        left = self.make_segment(1, score_dict)
        best = self.make_segment(2, score_dict)
        worse = self.make_segment(3, score_dict)
        root = self.make_segment(4, score_dict)
        root.pic_connection_matrix = np.asarray(
            [
                [left, 0],
                [0, 0],
            ],
            dtype=object,
        )
        root.binary_connection_matrix = (root.pic_connection_matrix != 0).astype(int)
        score_dict[left.piece_number, solver.JoinDirection.RIGHT, best.piece_number] = 1
        score_dict[left.piece_number, solver.JoinDirection.RIGHT, worse.piece_number] = 9

        filled = solver.fillHoles(root, [worse, best])

        self.assertEqual(1, filled)
        self.assertIs(root.pic_connection_matrix[0, 1], best)

    def test_trim_and_fill_uses_leftover_components_as_candidates(self):
        score_dict = {}
        left = self.make_segment(1, score_dict)
        candidate = self.make_segment(2, score_dict)
        root = self.make_segment(3, score_dict)
        root.pic_connection_matrix = np.asarray([[left, 0]], dtype=object)
        root.binary_connection_matrix = (root.pic_connection_matrix != 0).astype(int)
        leftover = candidate
        leftover.pic_connection_matrix = np.asarray([[candidate]], dtype=object)
        leftover.binary_connection_matrix = np.asarray([[1]])
        score_dict[left.piece_number, solver.JoinDirection.RIGHT, candidate.piece_number] = 1
        segments = [root, leftover]

        final_connection = solver.trimAndFillAssembly(segments, show_progress=False)

        self.assertEqual([root], segments)
        self.assertIs(final_connection.own_segment, root)
        self.assertIs(root.pic_connection_matrix[0, 1], candidate)


class ConnectionTests(unittest.TestCase):
    def make_pair_for_best_buddy_tests(self):
        score_dict = {}
        first = solver.Segment(
            np.zeros((2, 2, 3)),
            max_width=2,
            max_height=2,
            piece_number=1,
            component_id=1,
            score_dict=score_dict,
            gist=None,
            connections_dict={},
        )
        second = solver.Segment(
            np.ones((2, 2, 3)),
            max_width=2,
            max_height=2,
            piece_number=2,
            component_id=2,
            score_dict=score_dict,
            gist=None,
            connections_dict={},
        )
        first.calculateScoreEuclidean(second)
        return [first, second]

    def test_connect_best_buds_progress_output_is_descriptive(self):
        segments = self.make_pair_for_best_buddy_tests()

        with contextlib.redirect_stdout(io.StringIO()) as output:
            solver.connectBestBudsFirst(segments, original_size=2)

        self.assertIn("Best-buddy check:", output.getvalue())
        self.assertIn("->", output.getvalue())

    def test_connect_best_buds_can_run_without_progress_output(self):
        segments = self.make_pair_for_best_buddy_tests()

        with contextlib.redirect_stdout(io.StringIO()) as output:
            solver.connectBestBudsFirst(
                segments, original_size=2, show_progress=False)

        self.assertEqual("", output.getvalue())

    def test_connect_best_buds_reuses_best_match_scan(self):
        calls = []

        class FakeSegment:
            def __init__(self, piece_number):
                self.piece_number = piece_number
                self.best_connection_found_so_far = solver.BestConnection()

            def calculateConnectionsKruskal(self, other, boost):
                calls.append((self.piece_number, other.piece_number))
                connection = solver.BestConnection()
                connection.own_segment = self
                connection.join_segment = other
                connection.score = scores[self.piece_number, other.piece_number]
                return connection

        first = FakeSegment(1)
        second = FakeSegment(2)
        third = FakeSegment(3)
        scores = {
            (1, 2): 1,
            (1, 3): 2,
            (2, 3): 1,
            (2, 1): 2,
            (3, 1): 1,
            (3, 2): 2,
        }

        solver.connectBestBudsFirst(
            [first, second, third],
            original_size=3,
            show_progress=False,
        )

        self.assertEqual(
            sorted([(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]),
            sorted(calls),
        )

    def test_single_piece_best_buddy_connection_uses_best_edge_score(self):
        score_dict = {}
        first = solver.Segment(
            np.zeros((2, 2, 3)),
            max_width=2,
            max_height=2,
            piece_number=1,
            component_id=1,
            score_dict=score_dict,
            gist=None,
            connections_dict={},
        )
        second = solver.Segment(
            np.ones((2, 2, 3)),
            max_width=2,
            max_height=2,
            piece_number=2,
            component_id=2,
            score_dict=score_dict,
            gist=None,
            connections_dict={},
        )
        for direction, score in {
            solver.JoinDirection.UP: 9,
            solver.JoinDirection.DOWN: 8,
            solver.JoinDirection.LEFT: 1,
            solver.JoinDirection.RIGHT: 7,
        }.items():
            score_dict[1, direction, 2] = score

        connection = solver.calculateSinglePieceConnection(first, second)

        self.assertEqual(1, connection.score)
        self.assertEqual(7, connection.second_best_score)
        self.assertIs(connection.own_segment, first)
        self.assertIs(connection.join_segment, second)
        np.testing.assert_array_equal(
            np.asarray([[1, 1]]),
            connection.binary_connection_matrix,
        )
        self.assertIs(connection.pic_connection_matrix[0, 0], second)
        self.assertIs(connection.pic_connection_matrix[0, 1], first)

    def test_best_connection_strips_empty_rows_and_columns(self):
        connection = solver.BestConnection(
            pic_connection_matrix=np.asarray(
                [
                    [0, 0, 0],
                    [0, "piece", 0],
                    [0, 0, 0],
                ],
                dtype=object,
            ),
            binary_connection_matrix=np.asarray(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ]
            ),
        )

        connection.stripZeros()

        self.assertEqual((1, 1), connection.pic_connection_matrix.shape)
        self.assertEqual((1, 1), connection.binary_connection_matrix.shape)
        self.assertEqual("piece", connection.pic_connection_matrix[0, 0])
        self.assertEqual(1, connection.binary_connection_matrix[0, 0])

    def test_join_pieces_updates_owner_and_removes_joined_segment(self):
        score_dict = {}
        first = solver.Segment(
            np.zeros((2, 2, 3)),
            max_width=2,
            max_height=2,
            piece_number=1,
            component_id=1,
            score_dict=score_dict,
            gist=None,
            connections_dict={},
        )
        second = solver.Segment(
            np.ones((2, 2, 3)),
            max_width=2,
            max_height=2,
            piece_number=2,
            component_id=2,
            score_dict=score_dict,
            gist=None,
            connections_dict={},
        )
        connection = solver.BestConnection(
            own_segment=first,
            join_segment=second,
            pic_connection_matrix=np.asarray([[first, second]], dtype=object),
            binary_connection_matrix=np.asarray([[1, 1]]),
        )
        segments = [first, second]

        solver.joinPieces(connection, segments, original_size=2)

        self.assertEqual([first], segments)
        self.assertEqual(3, first.component_id)
        np.testing.assert_array_equal(np.asarray([[1, 1]]), first.binary_connection_matrix)
        self.assertIs(first.pic_connection_matrix[0, 0], first)
        self.assertIs(first.pic_connection_matrix[0, 1], second)


class KruskalAssemblyTests(unittest.TestCase):
    def make_segments(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, size=(12, 12, 3), dtype=np.uint8)
        segments = solver.breakUpImage(
            image,
            length=4,
            save_segments=False,
            color_type=solver.ColorType.RGB,
            score_algorithm=solver.ScoreAlgorithm.EUCLIDEAN,
        )
        solver.calculateScores(
            segments,
            solver.ScoreAlgorithm.EUCLIDEAN,
            show_progress=False,
            max_workers=1,
            executor_type="serial",
        )
        return segments, len(segments)

    def scan_assembly_history(self, segments, original_size):
        history = []
        while len(segments) > 1:
            best_connection = solver.findBestConnectionKruskal(
                segments,
                solver.CompareWithOtherSegments.ONLY_BEST,
                boost_priority_of_big_pieces_joining=False,
                compare_mode=solver.CompareWithOtherSegments.ONLY_BEST,
            )
            if best_connection.pic_connection_matrix is None:
                break
            history.append((
                best_connection.score,
                best_connection.own_segment.piece_number,
                best_connection.join_segment.piece_number,
            ))
            solver.joinPieces(best_connection, segments, original_size)
        return history

    def test_priority_queue_matches_full_scan_assembly(self):
        scan_segments, scan_original_size = self.make_segments()
        queue_segments, queue_original_size = self.make_segments()

        scan_history = self.scan_assembly_history(
            scan_segments, scan_original_size)
        queue_history = []

        def record_join(best_connection, round_number):
            queue_history.append((
                best_connection.score,
                best_connection.own_segment.piece_number,
                best_connection.join_segment.piece_number,
            ))

        queue_rounds = solver.assembleKruskalWithPriorityQueue(
            queue_segments,
            queue_original_size,
            on_join=record_join,
        )

        self.assertEqual(scan_history, queue_history)
        self.assertEqual(len(scan_history), queue_rounds)
        self.assertEqual(len(scan_segments), len(queue_segments))


if __name__ == "__main__":
    unittest.main()

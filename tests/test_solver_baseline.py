import contextlib
import io
import math
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Install requirements.txt to run solver tests") from exc


ROOT = Path(__file__).resolve().parents[1]
SOLVER_DIR = ROOT / "Python3"


def load_solver():
    sys.path.insert(0, str(SOLVER_DIR))
    try:
        import Solver
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest("Install requirements.txt to run solver tests") from exc
    return Solver


solver = load_solver()


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
            colortype=solver.ColorType.RGB,
            score_algorithum=solver.ScoreAlgorithum.EUCLIDEAN,
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
                    colortype=solver.ColorType.RGB,
                    score_algorithum=solver.ScoreAlgorithum.EUCLIDEAN,
                )

    def test_rejects_images_not_evenly_divisible_by_tile_size(self):
        image = np.zeros((5, 5, 3))

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                solver.breakUpImage(
                    image,
                    length=2,
                    save_segments=False,
                    colortype=solver.ColorType.RGB,
                    score_algorithum=solver.ScoreAlgorithum.EUCLIDEAN,
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
            myownNumber=piece_number,
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

        first.calculateScoreMahalonbis(second)

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

        first.calculateScoreEuclideanAndMahalonbis(second)

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
                solver.ScoreAlgorithum.EUCLIDEAN,
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
            solver.ScoreAlgorithum.EUCLIDEAN_AND_MAHALANOBIS,
            show_progress=False,
            max_workers=1,
            executor_type="serial",
        )
        solver.calculateScores(
            threaded_segments,
            solver.ScoreAlgorithum.EUCLIDEAN_AND_MAHALANOBIS,
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
            solver.ScoreAlgorithum.EUCLIDEAN_AND_MAHALANOBIS,
            show_progress=False,
            max_workers=1,
            executor_type="serial",
        )
        try:
            solver.calculateScores(
                process_segments,
                solver.ScoreAlgorithum.EUCLIDEAN_AND_MAHALANOBIS,
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
            solver.ScoreAlgorithum.EUCLIDEAN_AND_MAHALANOBIS,
        )

        self.assertEqual(0.0, segment.score_dict[("low",)])
        self.assertEqual(1.0, segment.score_dict[("mid",)])
        self.assertEqual(2.0, segment.score_dict[("high",)])


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


class ConnectionTests(unittest.TestCase):
    def make_pair_for_best_buddy_tests(self):
        score_dict = {}
        first = solver.Segment(
            np.zeros((2, 2, 3)),
            max_width=2,
            max_height=2,
            piece_number=1,
            myownNumber=1,
            score_dict=score_dict,
            gist=None,
            connections_dict={},
        )
        second = solver.Segment(
            np.ones((2, 2, 3)),
            max_width=2,
            max_height=2,
            piece_number=2,
            myownNumber=2,
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
            [first, second, third], original_size=3, show_progress=False)

        self.assertEqual(
            sorted([(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]),
            sorted(calls),
        )

    def test_best_connection_strips_empty_rows_and_columns(self):
        connection = solver.BestConnection(
            pic_connection_matix=np.asarray(
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

        self.assertEqual((1, 1), connection.pic_connection_matix.shape)
        self.assertEqual((1, 1), connection.binary_connection_matrix.shape)
        self.assertEqual("piece", connection.pic_connection_matix[0, 0])
        self.assertEqual(1, connection.binary_connection_matrix[0, 0])

    def test_join_pieces_updates_owner_and_removes_joined_segment(self):
        score_dict = {}
        first = solver.Segment(
            np.zeros((2, 2, 3)),
            max_width=2,
            max_height=2,
            piece_number=1,
            myownNumber=1,
            score_dict=score_dict,
            gist=None,
            connections_dict={},
        )
        second = solver.Segment(
            np.ones((2, 2, 3)),
            max_width=2,
            max_height=2,
            piece_number=2,
            myownNumber=2,
            score_dict=score_dict,
            gist=None,
            connections_dict={},
        )
        connection = solver.BestConnection(
            own_segment=first,
            join_segment=second,
            pic_connection_matix=np.asarray([[first, second]], dtype=object),
            binary_connection_matrix=np.asarray([[1, 1]]),
        )
        segments = [first, second]

        solver.joinPieces(connection, segments, original_size=2)

        self.assertEqual([first], segments)
        self.assertEqual(3, first.myownNumber)
        np.testing.assert_array_equal(np.asarray([[1, 1]]), first.binary_connection_matrix)
        self.assertIs(first.pic_connection_matix[0, 0], first)
        self.assertIs(first.pic_connection_matix[0, 1], second)


if __name__ == "__main__":
    unittest.main()

class DisjointSet:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def add(self, item):
        if item in self.parent:
            return
        self.parent[item] = item
        self.size[item] = 1

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            parent = self.find(parent)
            self.parent[item] = parent
        return parent

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]

    def largestSize(self):
        if not self.parent:
            return 0
        return max(
            self.size[root]
            for item, root in self.parent.items()
            if item == root
        )

    def groups(self):
        groups = {}
        for item in self.parent:
            root = self.find(item)
            groups.setdefault(root, []).append(item)
        return groups


def piecePosition(piece_number, width):
    return divmod(piece_number - 1, width)


def pieceCount(segment_list):
    if not segment_list:
        return 0
    return segment_list[0].max_height * segment_list[0].max_width


def possibleNeighborCount(segment_list):
    if not segment_list:
        return 0
    frame_height = segment_list[0].max_height
    frame_width = segment_list[0].max_width
    return (
        frame_height * (frame_width - 1)
        + frame_width * (frame_height - 1)
    )


def iterPlacedPieces(segment_list):
    seen = set()
    for segment in segment_list:
        matrix = segment.pic_connection_matrix
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                piece = matrix[row, col]
                if piece == 0:
                    continue
                if piece.piece_number in seen:
                    continue
                seen.add(piece.piece_number)
                yield segment, row, col, piece


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
    possible = possibleNeighborCount(segment_list)
    adjacent = 0
    correct = 0
    correct_components = DisjointSet()
    for _segment, _row, _col, piece in iterPlacedPieces(segment_list):
        correct_components.add(piece.piece_number)

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
                    neighbor = matrix[row, col + 1]
                    if isCorrectNeighbor(piece, neighbor, 0, 1):
                        correct += 1
                        correct_components.union(
                            piece.piece_number,
                            neighbor.piece_number,
                        )
                if row + 1 < rows and matrix[row + 1, col] != 0:
                    adjacent += 1
                    neighbor = matrix[row + 1, col]
                    if isCorrectNeighbor(piece, neighbor, 1, 0):
                        correct += 1
                        correct_components.union(
                            piece.piece_number,
                            neighbor.piece_number,
                        )

    incorrect = adjacent - correct
    return {
        "adjacent": adjacent,
        "correct": correct,
        "incorrect": incorrect,
        "possible": possible,
        "coverage": correct / possible if possible else 0.0,
        "precision": correct / adjacent if adjacent else 0.0,
        "largest_correct_component": correct_components.largestSize(),
    }


def componentSize(segment):
    return int((segment.binary_connection_matrix != 0).sum())


def largestAssembledComponentSize(segment_list):
    if not segment_list:
        return 0
    return max(componentSize(segment) for segment in segment_list)


def directPlacementQuality(segment_list):
    total = pieceCount(segment_list)
    best_direct = 0
    best_offset = None
    for segment in segment_list:
        votes = {}
        for _segment, row, col, piece in iterPlacedPieces([segment]):
            true_row, true_col = piecePosition(piece.piece_number, piece.max_height)
            offset = (true_row - row, true_col - col)
            votes[offset] = votes.get(offset, 0) + 1
        if not votes:
            continue
        offset, direct = max(votes.items(), key=lambda item: item[1])
        if direct > best_direct:
            best_direct = direct
            best_offset = offset
    return {
        "direct": best_direct,
        "possible": total,
        "accuracy": best_direct / total if total else 0.0,
        "offset": best_offset,
    }


def scoreValue(score_dict, own_number, direction, join_number):
    if hasattr(score_dict, "scoreValue"):
        return score_dict.scoreValue(own_number, direction, join_number)
    return score_dict.get((own_number, direction, join_number), float("inf"))


def assembledNeighborPairs(segment_list):
    from .enums import JoinDirection

    directions = (
        (0, 1, JoinDirection.RIGHT),
        (1, 0, JoinDirection.DOWN),
    )
    for segment_index, segment in enumerate(segment_list):
        matrix = segment.pic_connection_matrix
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                piece = matrix[row, col]
                if piece == 0:
                    continue
                for row_delta, col_delta, direction in directions:
                    neighbor_row = row + row_delta
                    neighbor_col = col + col_delta
                    if not (
                            0 <= neighbor_row < rows
                            and 0 <= neighbor_col < cols):
                        continue
                    neighbor = matrix[neighbor_row, neighbor_col]
                    if neighbor == 0:
                        continue
                    yield {
                        "segment_index": segment_index,
                        "piece": piece,
                        "neighbor": neighbor,
                        "row": row,
                        "col": col,
                        "neighbor_row": neighbor_row,
                        "neighbor_col": neighbor_col,
                        "direction": direction,
                        "row_delta": row_delta,
                        "col_delta": col_delta,
                    }


def correctIslandDetails(segment_list, limit=5):
    islands = DisjointSet()
    positions = {}
    segment_index_by_segment = {
        segment: index
        for index, segment in enumerate(segment_list)
    }
    for segment, row, col, piece in iterPlacedPieces(segment_list):
        islands.add(piece.piece_number)
        true_row, true_col = piecePosition(piece.piece_number, piece.max_height)
        positions[piece.piece_number] = {
            "segment_index": segment_index_by_segment[segment],
            "row": row,
            "col": col,
            "true_row": true_row,
            "true_col": true_col,
        }

    for pair in assembledNeighborPairs(segment_list):
        piece = pair["piece"]
        neighbor = pair["neighbor"]
        if isCorrectNeighbor(
                piece,
                neighbor,
                pair["row_delta"],
                pair["col_delta"]):
            islands.union(piece.piece_number, neighbor.piece_number)

    details = []
    for pieces in islands.groups().values():
        piece_positions = [
            positions[piece_number]
            for piece_number in pieces
            if piece_number in positions
        ]
        true_rows = [position["true_row"] for position in piece_positions]
        true_cols = [position["true_col"] for position in piece_positions]
        placed_rows = [position["row"] for position in piece_positions]
        placed_cols = [position["col"] for position in piece_positions]
        segment_indices = sorted({
            position["segment_index"]
            for position in piece_positions
        })
        details.append({
            "size": len(pieces),
            "sample_pieces": tuple(sorted(pieces)[:8]),
            "segment_indices": tuple(segment_indices),
            "true_bbox": (
                min(true_rows),
                min(true_cols),
                max(true_rows),
                max(true_cols),
            ),
            "placed_bbox": (
                min(placed_rows),
                min(placed_cols),
                max(placed_rows),
                max(placed_cols),
            ),
        })
    details.sort(key=lambda item: (-item["size"], item["sample_pieces"]))
    return details[:limit]


def bestDirectPlacementDetails(segment_list):
    best = None
    for segment_index, segment in enumerate(segment_list):
        votes = {}
        piece_data = []
        for _segment, row, col, piece in iterPlacedPieces([segment]):
            true_row, true_col = piecePosition(piece.piece_number, piece.max_height)
            offset = (true_row - row, true_col - col)
            votes[offset] = votes.get(offset, 0) + 1
            piece_data.append((piece.piece_number, row, col, true_row, true_col))
        if not votes:
            continue
        offset, direct = max(votes.items(), key=lambda item: item[1])
        if best is None or direct > best["direct"]:
            mismatches = []
            for piece_number, row, col, true_row, true_col in piece_data:
                if (true_row - row, true_col - col) == offset:
                    continue
                mismatches.append({
                    "piece": piece_number,
                    "placed": (row, col),
                    "true": (true_row, true_col),
                    "offset": (true_row - row, true_col - col),
                })
            best = {
                "segment_index": segment_index,
                "segment_size": componentSize(segment),
                "offset": offset,
                "direct": direct,
                "mismatch_count": len(mismatches),
                "mismatches": sorted(
                    mismatches,
                    key=lambda item: item["piece"],
                )[:10],
            }
    return best


def falseSeamDetails(segment_list, limit=12):
    if not segment_list:
        return []
    score_dict = segment_list[0].score_dict
    false_seams = []
    for pair in assembledNeighborPairs(segment_list):
        piece = pair["piece"]
        neighbor = pair["neighbor"]
        if isCorrectNeighbor(
                piece,
                neighbor,
                pair["row_delta"],
                pair["col_delta"]):
            continue
        piece_true = piecePosition(piece.piece_number, piece.max_height)
        neighbor_true = piecePosition(
            neighbor.piece_number,
            neighbor.max_height,
        )
        score = scoreValue(
            score_dict,
            piece.piece_number,
            pair["direction"],
            neighbor.piece_number,
        )
        false_seams.append({
            "score": float(score),
            "piece": piece.piece_number,
            "neighbor": neighbor.piece_number,
            "direction": pair["direction"].name.lower(),
            "segment_index": pair["segment_index"],
            "placed": (pair["row"], pair["col"]),
            "neighbor_placed": (pair["neighbor_row"], pair["neighbor_col"]),
            "true": piece_true,
            "neighbor_true": neighbor_true,
            "true_delta": (
                neighbor_true[0] - piece_true[0],
                neighbor_true[1] - piece_true[1],
            ),
        })
    false_seams.sort(key=lambda item: (item["score"], item["piece"]))
    return false_seams[:limit]


def errorDiagnosticReport(segment_list, limit=12):
    return {
        "quality": paperStyleReport(segment_list),
        "best_direct_component": bestDirectPlacementDetails(segment_list),
        "correct_islands": correctIslandDetails(segment_list, limit=5),
        "false_seams": falseSeamDetails(segment_list, limit=limit),
    }


def formatErrorDiagnosticReport(report):
    lines = ["Error diagnostics:"]
    best_direct = report["best_direct_component"]
    if best_direct is None:
        lines.append("  best direct component: none")
    else:
        lines.append(
            "  best direct component: "
            f"segment={best_direct['segment_index']} "
            f"size={best_direct['segment_size']} "
            f"direct={best_direct['direct']} "
            f"offset={best_direct['offset']} "
            f"mismatches={best_direct['mismatch_count']}"
        )
        if best_direct["mismatches"]:
            lines.append("  sample direct mismatches:")
            for mismatch in best_direct["mismatches"]:
                lines.append(
                    "    "
                    f"piece={mismatch['piece']} "
                    f"placed={mismatch['placed']} "
                    f"true={mismatch['true']} "
                    f"offset={mismatch['offset']}"
                )

    lines.append("  largest correct islands:")
    for index, island in enumerate(report["correct_islands"], start=1):
        lines.append(
            "    "
            f"{index}. size={island['size']} "
            f"segments={island['segment_indices']} "
            f"true_bbox={island['true_bbox']} "
            f"placed_bbox={island['placed_bbox']} "
            f"sample={island['sample_pieces']}"
        )

    lines.append("  lowest-score false seams:")
    if not report["false_seams"]:
        lines.append("    none")
    for seam in report["false_seams"]:
        lines.append(
            "    "
            f"score={seam['score']:.6g} "
            f"{seam['piece']}->{seam['neighbor']} "
            f"dir={seam['direction']} "
            f"placed={seam['placed']}->{seam['neighbor_placed']} "
            f"true={seam['true']}->{seam['neighbor_true']} "
            f"true_delta={seam['true_delta']}"
        )
    return "\n".join(lines)


def trueNeighborRankStats(segment_list, max_rank=5):
    if not segment_list:
        return {
            "total": 0,
            "top1": 0,
            "top2": 0,
            "top5": 0,
            "top1_accuracy": 0.0,
            "top2_accuracy": 0.0,
            "top5_accuracy": 0.0,
        }

    from .enums import JoinDirection

    width = segment_list[0].max_width
    piece_count = width * segment_list[0].max_height
    score_dict = segment_list[0].score_dict
    if hasattr(score_dict, "scoreValue"):
        score_value = score_dict.scoreValue
    else:
        def score_value(own_number, direction, join_number):
            return score_dict[own_number, direction, join_number]
    directions = (
        (JoinDirection.RIGHT, 0, 1),
        (JoinDirection.DOWN, 1, 0),
        (JoinDirection.LEFT, 0, -1),
        (JoinDirection.UP, -1, 0),
    )

    total = 0
    top1 = 0
    top2 = 0
    top5 = 0
    for piece_number in range(1, piece_count + 1):
        row, col = piecePosition(piece_number, width)
        for direction, row_delta, col_delta in directions:
            neighbor_row = row + row_delta
            neighbor_col = col + col_delta
            if not (0 <= neighbor_row < width and 0 <= neighbor_col < width):
                continue
            true_neighbor = neighbor_row * width + neighbor_col + 1
            scores = []
            for candidate in range(1, piece_count + 1):
                if candidate == piece_number:
                    continue
                scores.append((
                    score_value(piece_number, direction, candidate),
                    candidate,
                ))
            scores.sort()
            rank = max_rank + 1
            for index, (_score, candidate) in enumerate(scores[:max_rank]):
                if candidate == true_neighbor:
                    rank = index + 1
                    break
            total += 1
            top1 += rank <= 1
            top2 += rank <= 2
            top5 += rank <= 5

    return {
        "total": total,
        "top1": top1,
        "top2": top2,
        "top5": top5,
        "top1_accuracy": top1 / total if total else 0.0,
        "top2_accuracy": top2 / total if total else 0.0,
        "top5_accuracy": top5 / total if total else 0.0,
    }


def paperStyleReport(segment_list, include_rank_stats=False):
    neighbors = assemblyQuality(segment_list)
    direct = directPlacementQuality(segment_list)
    report = {
        "pieces": pieceCount(segment_list),
        "components": len(segment_list),
        "largest_component": largestAssembledComponentSize(segment_list),
        "neighbor": neighbors,
        "direct": direct,
    }
    if include_rank_stats:
        report["true_neighbor_rank"] = trueNeighborRankStats(segment_list)
    return report


def formatPaperStyleReport(report):
    neighbor = report["neighbor"]
    direct = report["direct"]
    lines = [
        "Paper-style quality:",
        (
            "  components="
            f"{report['components']} largest_component="
            f"{report['largest_component']}/{report['pieces']}"
        ),
        (
            "  neighbors: adjacent="
            f"{neighbor['adjacent']} correct={neighbor['correct']} "
            f"incorrect={neighbor['incorrect']} possible={neighbor['possible']} "
            f"coverage={neighbor['coverage']:.3f} "
            f"precision={neighbor['precision']:.3f}"
        ),
        (
            "  largest_correct_component="
            f"{neighbor['largest_correct_component']}/{report['pieces']}"
        ),
        (
            "  direct: correct="
            f"{direct['direct']}/{direct['possible']} "
            f"accuracy={direct['accuracy']:.3f} offset={direct['offset']}"
        ),
    ]
    rank = report.get("true_neighbor_rank")
    if rank is not None:
        lines.append(
            "  true-neighbor rank: "
            f"top1={rank['top1_accuracy']:.3f} "
            f"top2={rank['top2_accuracy']:.3f} "
            f"top5={rank['top5_accuracy']:.3f} "
            f"total={rank['total']}"
        )
    return "\n".join(lines)

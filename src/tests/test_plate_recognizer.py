import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from core.plate_recognizer import PlateCandidate, fuse_plate_candidates, summarize_plate_candidates


def test_plate_candidate_fusion_prefers_consistent_text():
    result = fuse_plate_candidates(
        [
            PlateCandidate(text="abc123", confidence=0.82, frame_index=3),
            PlateCandidate(text="ABC123", confidence=0.91, frame_index=5),
            PlateCandidate(text="ABD123", confidence=0.60, frame_index=7),
        ]
    )

    assert result is not None
    assert result.text == "ABC123"
    assert result.support_count == 2
    assert result.source_frame_indices == [3, 5]


def test_summarize_plate_candidates_keeps_chinese_prefix_and_best_type():
    summary = summarize_plate_candidates(
        [
            PlateCandidate(text="京A12345", confidence=0.72, frame_index=1, plate_type="blue", plate_type_id=0),
            PlateCandidate(text="京A12345", confidence=0.95, frame_index=2, plate_type="blue", plate_type_id=0),
            PlateCandidate(text="京A12346", confidence=0.61, frame_index=3, plate_type="blue", plate_type_id=0),
        ],
        min_confidence=0.6,
    )

    assert summary is not None
    assert summary["plate_text"] == "京A12345"
    assert summary["plate_support_count"] == 2
    assert summary["plate_type"] == "blue"
    assert summary["plate_source_frame_indices"] == [1, 2]

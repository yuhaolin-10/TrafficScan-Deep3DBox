import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from core.traffic_counter import CountLineRule, TrafficCounter


def test_traffic_counter_counts_crossing_once_per_track():
    counter = TrafficCounter(CountLineRule(name="line_01", start=(0.0, 10.0), end=(20.0, 10.0), direction_mode="forward"))

    event = counter.update("track-1", (5.0, 8.0), (5.0, 12.0), frame_index=4)
    duplicate = counter.update("track-1", (5.0, 12.0), (5.0, 16.0), frame_index=5)

    assert event is not None
    assert event.direction == "forward"
    assert event.frame_index == 4
    assert duplicate is None
    assert counter.counted_total() == 1

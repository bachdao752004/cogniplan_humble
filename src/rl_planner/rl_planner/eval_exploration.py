#!/usr/bin/env python3
"""Evaluate exploration metrics from a ROS2 bag (.db3).

Metrics:
- coverage (%) over time, based on projected map vs ground-truth map
- time-to-X% coverage (default: 70/80/90)
- traveled distance from /state_estimation
- waypoint publish rate from /way_point
- planner runtime stats from /runtime
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import Float32


@dataclass
class TopicMeta:
    topic_id: int
    name: str
    type_name: str


def read_topics(conn: sqlite3.Connection) -> Dict[str, TopicMeta]:
    rows = conn.execute("SELECT id, name, type FROM topics").fetchall()
    return {name: TopicMeta(topic_id=topic_id, name=name, type_name=type_name) for topic_id, name, type_name in rows}


def iter_topic_messages(
    conn: sqlite3.Connection,
    topics: Dict[str, TopicMeta],
    topic_names: Sequence[str],
) -> Iterable[Tuple[int, str, object]]:
    wanted = [topics[name] for name in topic_names if name in topics]
    if not wanted:
        return []
    ids = ",".join(str(t.topic_id) for t in wanted)
    id_to_meta = {t.topic_id: t for t in wanted}
    rows = conn.execute(
        f"SELECT topic_id, timestamp, data FROM messages WHERE topic_id IN ({ids}) ORDER BY timestamp ASC"
    )
    for topic_id, ts, blob in rows:
        meta = id_to_meta[topic_id]
        msg_type = get_message(meta.type_name)
        yield ts, meta.name, deserialize_message(blob, msg_type)


def free_cells(msg: OccupancyGrid) -> Set[Tuple[int, int]]:
    width = msg.info.width
    free = set()
    for idx, value in enumerate(msg.data):
        if value == 0:
            y, x = divmod(idx, width)
            free.add((x, y))
    return free


def cell_to_world(x: int, y: int, msg: OccupancyGrid) -> Tuple[float, float]:
    ox = msg.info.origin.position.x
    oy = msg.info.origin.position.y
    res = msg.info.resolution
    return ox + x * res, oy + y * res


def world_to_cell(wx: float, wy: float, msg: OccupancyGrid) -> Optional[Tuple[int, int]]:
    ox = msg.info.origin.position.x
    oy = msg.info.origin.position.y
    res = msg.info.resolution
    x = int(round((wx - ox) / res))
    y = int(round((wy - oy) / res))
    if 0 <= x < msg.info.width and 0 <= y < msg.info.height:
        return x, y
    return None


def projected_to_gt_free_cells(projected: OccupancyGrid, gt: OccupancyGrid, gt_free: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    covered = set()
    for px, py in free_cells(projected):
        wx, wy = cell_to_world(px, py, projected)
        gcell = world_to_cell(wx, wy, gt)
        if gcell is not None and gcell in gt_free:
            covered.add(gcell)
    return covered


def compute_path_length(odom_msgs: List[Tuple[int, Odometry]]) -> float:
    if len(odom_msgs) < 2:
        return 0.0
    dist = 0.0
    prev = odom_msgs[0][1].pose.pose.position
    for _, msg in odom_msgs[1:]:
        cur = msg.pose.pose.position
        dist += math.hypot(cur.x - prev.x, cur.y - prev.y)
        prev = cur
    return dist


def compute_hz(stamps_ns: List[int]) -> float:
    if len(stamps_ns) < 2:
        return 0.0
    duration = (stamps_ns[-1] - stamps_ns[0]) / 1e9
    if duration <= 0:
        return 0.0
    return (len(stamps_ns) - 1) / duration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Path to rosbag .db3 file")
    parser.add_argument("--ground-truth-topic", default="/ground_truth_map", help="OccupancyGrid ground truth topic")
    parser.add_argument("--projected-topic", default="/projected_map", help="Observed OccupancyGrid topic")
    parser.add_argument("--odom-topic", default="/state_estimation")
    parser.add_argument("--runtime-topic", default="/runtime")
    parser.add_argument("--waypoint-topic", default="/way_point")
    parser.add_argument("--thresholds", default="70,80,90", help="Coverage thresholds in percent")
    parser.add_argument("--out-prefix", default="exploration_eval", help="Output prefix for CSV/JSON")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    conn = sqlite3.connect(str(bag_path))
    try:
        topics = read_topics(conn)
        needed = [
            args.ground_truth_topic,
            args.projected_topic,
            args.odom_topic,
            args.runtime_topic,
            args.waypoint_topic,
        ]

        gt_msg: Optional[OccupancyGrid] = None
        projected_msgs: List[Tuple[int, OccupancyGrid]] = []
        odom_msgs: List[Tuple[int, Odometry]] = []
        runtime_vals: List[float] = []
        waypoint_stamps: List[int] = []

        for ts, topic, msg in iter_topic_messages(conn, topics, needed):
            if topic == args.ground_truth_topic and isinstance(msg, OccupancyGrid):
                if gt_msg is None:
                    gt_msg = msg
            elif topic == args.projected_topic and isinstance(msg, OccupancyGrid):
                projected_msgs.append((ts, msg))
            elif topic == args.odom_topic and isinstance(msg, Odometry):
                odom_msgs.append((ts, msg))
            elif topic == args.runtime_topic and isinstance(msg, Float32):
                runtime_vals.append(float(msg.data))
            elif topic == args.waypoint_topic and isinstance(msg, PointStamped):
                waypoint_stamps.append(ts)

        if gt_msg is None:
            raise SystemExit(f"Ground truth topic '{args.ground_truth_topic}' not found in bag.")
        if not projected_msgs:
            raise SystemExit(f"Projected map topic '{args.projected_topic}' not found in bag.")

        gt_free = free_cells(gt_msg)
        total_gt_free = len(gt_free)
        covered: Set[Tuple[int, int]] = set()
        t0 = projected_msgs[0][0]

        rows = []
        for ts, pmsg in projected_msgs:
            covered.update(projected_to_gt_free_cells(pmsg, gt_msg, gt_free))
            coverage = 100.0 * (len(covered) / total_gt_free) if total_gt_free else 0.0
            elapsed = (ts - t0) / 1e9
            rows.append((elapsed, len(covered), total_gt_free, coverage))

        thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
        time_to = {}
        for thr in thresholds:
            hit = next((t for t, _, _, c in rows if c >= thr), None)
            time_to[f"time_to_{int(thr)}"] = hit

        path_length = compute_path_length(odom_msgs)
        waypoint_hz = compute_hz(waypoint_stamps)
        runtime_mean = sum(runtime_vals) / len(runtime_vals) if runtime_vals else 0.0
        runtime_max = max(runtime_vals) if runtime_vals else 0.0

        summary = {
            "bag": str(bag_path),
            "ground_truth_topic": args.ground_truth_topic,
            "projected_topic": args.projected_topic,
            "total_ground_truth_free_cells": total_gt_free,
            "final_covered_free_cells": len(covered),
            "final_coverage_percent": rows[-1][3],
            "path_length_m": path_length,
            "waypoint_hz": waypoint_hz,
            "planner_runtime_mean_s": runtime_mean,
            "planner_runtime_max_s": runtime_max,
            **time_to,
        }

        out_csv = Path(f"{args.out_prefix}_coverage.csv")
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["elapsed_time_s", "covered_free_cells", "total_free_cells", "coverage_percent"])
            writer.writerows(rows)

        out_json = Path(f"{args.out_prefix}_summary.json")
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(json.dumps(summary, indent=2))
        print(f"[OK] wrote {out_csv}")
        print(f"[OK] wrote {out_json}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

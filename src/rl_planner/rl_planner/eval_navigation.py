#!/usr/bin/env python3
"""Evaluate navigation metrics from a ROS2 bag (.db3).

Per-goal metrics:
- success (goal reached within timeout)
- time-to-goal
- path length while navigating to goal
- final goal error
- stability: waypoint oscillation count (A->B->A pattern)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


@dataclass
class TopicMeta:
    topic_id: int
    name: str
    type_name: str


@dataclass
class GoalEpisode:
    goal_time_ns: int
    goal_x: float
    goal_y: float
    success: bool
    success_time_s: Optional[float]
    path_length_m: float
    final_goal_error_m: float
    waypoint_count: int
    waypoint_oscillation_count: int


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


def euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def compute_path_length(odom_slice: List[Tuple[int, Odometry]]) -> float:
    if len(odom_slice) < 2:
        return 0.0
    dist = 0.0
    prev = odom_slice[0][1].pose.pose.position
    for _, msg in odom_slice[1:]:
        cur = msg.pose.pose.position
        dist += euclidean(prev.x, prev.y, cur.x, cur.y)
        prev = cur
    return dist


def waypoint_oscillation_count(points: List[Tuple[float, float]], eps: float = 0.2) -> int:
    if len(points) < 3:
        return 0
    osc = 0
    for i in range(2, len(points)):
        a = points[i - 2]
        b = points[i - 1]
        c = points[i]
        if euclidean(a[0], a[1], c[0], c[1]) < eps and euclidean(a[0], a[1], b[0], b[1]) >= eps:
            osc += 1
    return osc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Path to rosbag .db3 file")
    parser.add_argument("--goal-topic", default="/goal_pose")
    parser.add_argument("--odom-topic", default="/state_estimation")
    parser.add_argument("--waypoint-topic", default="/way_point")
    parser.add_argument("--goal-threshold", type=float, default=2.0, help="Goal reached threshold in meters")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout per goal in seconds")
    parser.add_argument("--out-prefix", default="navigation_eval", help="Output prefix for CSV/JSON")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    conn = sqlite3.connect(str(bag_path))
    try:
        topics = read_topics(conn)
        needed = [args.goal_topic, args.odom_topic, args.waypoint_topic]

        goals: List[Tuple[int, PoseStamped]] = []
        odom_msgs: List[Tuple[int, Odometry]] = []
        waypoints: List[Tuple[int, PointStamped]] = []

        for ts, topic, msg in iter_topic_messages(conn, topics, needed):
            if topic == args.goal_topic and isinstance(msg, PoseStamped):
                goals.append((ts, msg))
            elif topic == args.odom_topic and isinstance(msg, Odometry):
                odom_msgs.append((ts, msg))
            elif topic == args.waypoint_topic and isinstance(msg, PointStamped):
                waypoints.append((ts, msg))

        if not goals:
            raise SystemExit(f"No goals found on topic '{args.goal_topic}'.")
        if not odom_msgs:
            raise SystemExit(f"No odometry found on topic '{args.odom_topic}'.")

        episodes: List[GoalEpisode] = []
        timeout_ns = int(args.timeout * 1e9)

        for i, (gts, gmsg) in enumerate(goals):
            next_goal_ts = goals[i + 1][0] if i + 1 < len(goals) else 2**63 - 1
            end_ts = min(gts + timeout_ns, next_goal_ts)
            gx = float(gmsg.pose.position.x)
            gy = float(gmsg.pose.position.y)

            odom_slice = [(ts, msg) for ts, msg in odom_msgs if gts <= ts <= end_ts]
            if not odom_slice:
                continue

            reached_ts: Optional[int] = None
            final_pos = odom_slice[-1][1].pose.pose.position
            for ts, omsg in odom_slice:
                pos = omsg.pose.pose.position
                if euclidean(pos.x, pos.y, gx, gy) <= args.goal_threshold:
                    reached_ts = ts
                    final_pos = pos
                    break

            if reached_ts is not None:
                odom_used = [(ts, msg) for ts, msg in odom_slice if ts <= reached_ts]
            else:
                odom_used = odom_slice

            path_len = compute_path_length(odom_used)
            final_err = euclidean(final_pos.x, final_pos.y, gx, gy)
            success = reached_ts is not None
            t_goal = ((reached_ts - gts) / 1e9) if reached_ts is not None else None

            wp_slice = [(ts, msg) for ts, msg in waypoints if gts <= ts <= end_ts]
            wp_points = [(float(msg.point.x), float(msg.point.y)) for _, msg in wp_slice]
            osc = waypoint_oscillation_count(wp_points)

            episodes.append(
                GoalEpisode(
                    goal_time_ns=gts,
                    goal_x=gx,
                    goal_y=gy,
                    success=success,
                    success_time_s=t_goal,
                    path_length_m=path_len,
                    final_goal_error_m=final_err,
                    waypoint_count=len(wp_points),
                    waypoint_oscillation_count=osc,
                )
            )

        if not episodes:
            raise SystemExit("No valid episodes could be evaluated.")

        success_eps = [e for e in episodes if e.success]
        success_rate = 100.0 * len(success_eps) / len(episodes)
        mean_ttg = sum(e.success_time_s for e in success_eps if e.success_time_s is not None) / len(success_eps) if success_eps else None
        mean_path = sum(e.path_length_m for e in episodes) / len(episodes)
        mean_final_err = sum(e.final_goal_error_m for e in episodes) / len(episodes)
        mean_osc = sum(e.waypoint_oscillation_count for e in episodes) / len(episodes)

        summary = {
            "bag": str(bag_path),
            "episodes": len(episodes),
            "success_rate_percent": success_rate,
            "mean_time_to_goal_s": mean_ttg,
            "mean_path_length_m": mean_path,
            "mean_final_goal_error_m": mean_final_err,
            "mean_waypoint_oscillation_count": mean_osc,
            "goal_threshold_m": args.goal_threshold,
            "timeout_s": args.timeout,
        }

        out_csv = Path(f"{args.out_prefix}_episodes.csv")
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "goal_time_ns",
                    "goal_x",
                    "goal_y",
                    "success",
                    "success_time_s",
                    "path_length_m",
                    "final_goal_error_m",
                    "waypoint_count",
                    "waypoint_oscillation_count",
                ]
            )
            for e in episodes:
                writer.writerow(
                    [
                        e.goal_time_ns,
                        e.goal_x,
                        e.goal_y,
                        int(e.success),
                        e.success_time_s if e.success_time_s is not None else "",
                        e.path_length_m,
                        e.final_goal_error_m,
                        e.waypoint_count,
                        e.waypoint_oscillation_count,
                    ]
                )

        out_json = Path(f"{args.out_prefix}_summary.json")
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(json.dumps(summary, indent=2))
        print(f"[OK] wrote {out_csv}")
        print(f"[OK] wrote {out_json}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

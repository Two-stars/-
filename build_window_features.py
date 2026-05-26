"""Build well-wise 0.5 m window statistical features for train_val CSV files.

The script appends window features to:
    dataset/train_val/train.csv
    dataset/train_val/test.csv

It keeps row order, labels, and original columns unchanged, except that old
window feature columns are removed before recalculation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_DIR = "dataset/train_val"
TRAIN_NAME = "train.csv"
TEST_NAME = "test.csv"
TRAIN_BACKUP_NAME = "train_before_window0p5.csv"
TEST_BACKUP_NAME = "test_before_window0p5.csv"

WELL_COL = "井名"
DEPTH_COL = "深度"
WINDOW_NAME = "win0p5m"
COUNT_COL = f"{WINDOW_NAME}_count"

WINDOW_BASE_COLUMNS = [
    "GR",
    "SP",
    "AC",
    "CAL",
    "DEN",
    "CNL",
    "PE",
    "RD",
    "RD_RS_ratio",
    "SW",
]

MISSING_VALUES = [-9999, "-9999", "#REF!", "", " ", "NA", "N/A", "nan", "NaN", "None"]
WINDOW_SUFFIXES = (
    f"_{WINDOW_NAME}_mean",
    f"_{WINDOW_NAME}_std",
    f"_{WINDOW_NAME}_range",
    f"_{WINDOW_NAME}_slope",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 0.5 m well-wise window features.")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--window_half_width", type=float, default=0.5)
    parser.add_argument("--min_points", type=int, default=3)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", na_values=MISSING_VALUES, keep_default_na=True)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk", na_values=MISSING_VALUES, keep_default_na=True)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError as exc:
        raise PermissionError(
            f"无法写入 {path}。请先关闭正在打开该 CSV 的 Excel/WPS/编辑器后重试。"
        ) from exc


def backup_once(source: Path, backup: Path) -> bool:
    if backup.exists():
        return False
    shutil.copy2(source, backup)
    return True


def build_window_feature_names() -> Tuple[List[str], List[str]]:
    all_features: List[str] = []
    nosw_features: List[str] = []

    for col in WINDOW_BASE_COLUMNS:
        col_features = [
            f"{col}_{WINDOW_NAME}_mean",
            f"{col}_{WINDOW_NAME}_std",
            f"{col}_{WINDOW_NAME}_range",
            f"{col}_{WINDOW_NAME}_slope",
        ]
        all_features.extend(col_features)
        if col != "SW":
            nosw_features.extend(col_features)

    all_features.append(COUNT_COL)
    nosw_features.append(COUNT_COL)
    return all_features, nosw_features


def is_window_feature(col: str) -> bool:
    return col == COUNT_COL or col.endswith(WINDOW_SUFFIXES)


def remove_old_window_features(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    old_cols = [col for col in frame.columns if is_window_feature(col)]
    if old_cols:
        frame = frame.drop(columns=old_cols)
    return frame, old_cols


def to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric Series; missing columns become all-NaN."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    series = df[col].replace(MISSING_VALUES, np.nan)
    return pd.to_numeric(series, errors="coerce")


def validate_well_and_depth(frame: pd.DataFrame, path: Path) -> Tuple[pd.Series, pd.Series]:
    if WELL_COL not in frame.columns:
        raise ValueError(f"{path} 缺少井名列: {WELL_COL}")
    if DEPTH_COL not in frame.columns:
        raise ValueError(f"{path} 缺少深度列: {DEPTH_COL}")

    well = frame[WELL_COL]
    well_clean = well.astype("string").str.strip()
    missing_well = well.isna() | well_clean.isna() | well_clean.eq("")
    if missing_well.any():
        bad_rows = frame.index[missing_well].tolist()[:10]
        raise ValueError(f"{path} 存在空井名或缺失井名，示例行号: {bad_rows}")

    depth = to_numeric_series(frame, DEPTH_COL)
    missing_depth = depth.isna()
    if missing_depth.any():
        bad_rows = frame.index[missing_depth].tolist()[:10]
        raise ValueError(f"{path} 存在无法转为数值的深度，示例行号: {bad_rows}")

    return well_clean.astype(str), depth.astype(float)


def compute_slope(depth: np.ndarray, values: np.ndarray) -> float:
    valid = ~np.isnan(values)
    if int(valid.sum()) < 3:
        return np.nan

    x = depth[valid].astype(float)
    y = values[valid].astype(float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered**2))
    if denominator <= 0:
        return np.nan
    return float(np.sum(x_centered * (y - y.mean())) / denominator)


def compute_window_features(
    frame: pd.DataFrame,
    path: Path,
    window_half_width: float,
    min_points: int,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, object]]:
    frame = frame.reset_index(drop=True).copy()
    original_rows = len(frame)
    original_cols = len(frame.columns)
    frame, removed_old_cols = remove_old_window_features(frame)

    well_clean, depth_numeric = validate_well_and_depth(frame, path)
    numeric_columns = {col: to_numeric_series(frame, col).to_numpy(dtype=float) for col in WINDOW_BASE_COLUMNS}

    new_features_all, _ = build_window_feature_names()
    result = pd.DataFrame(index=frame.index)
    for feature in new_features_all:
        result[feature] = np.nan
    result[COUNT_COL] = np.nan

    work = pd.DataFrame(
        {
            "__row__": np.arange(len(frame), dtype=int),
            WELL_COL: well_clean.to_numpy(),
            "__depth_numeric__": depth_numeric.to_numpy(dtype=float),
        }
    )

    well_counts = work.groupby(WELL_COL, sort=True).size().astype(int).to_dict()

    for _, group in work.groupby(WELL_COL, sort=False):
        group = group.sort_values("__depth_numeric__", kind="mergesort")
        rows = group["__row__"].to_numpy(dtype=int)
        depths = group["__depth_numeric__"].to_numpy(dtype=float)
        values_by_col = {col: numeric_columns[col][rows] for col in WINDOW_BASE_COLUMNS}

        for sorted_pos, row_idx in enumerate(rows):
            center_depth = depths[sorted_pos]
            left = np.searchsorted(depths, center_depth - window_half_width, side="left")
            right = np.searchsorted(depths, center_depth + window_half_width, side="right")
            point_count = int(right - left)
            result.at[row_idx, COUNT_COL] = point_count

            if point_count < min_points:
                continue

            window_depth = depths[left:right]
            for col in WINDOW_BASE_COLUMNS:
                window_values = values_by_col[col][left:right]
                valid_values = window_values[~np.isnan(window_values)]
                if len(valid_values) < min_points:
                    continue

                result.at[row_idx, f"{col}_{WINDOW_NAME}_mean"] = float(np.mean(valid_values))
                result.at[row_idx, f"{col}_{WINDOW_NAME}_std"] = float(np.std(valid_values, ddof=1))
                result.at[row_idx, f"{col}_{WINDOW_NAME}_range"] = float(np.max(valid_values) - np.min(valid_values))
                result.at[row_idx, f"{col}_{WINDOW_NAME}_slope"] = compute_slope(window_depth, window_values)

    output = pd.concat([frame, result[new_features_all]], axis=1)
    if len(output) != original_rows:
        raise RuntimeError(f"{path} 输出行数异常: 原始 {original_rows}, 输出 {len(output)}")

    non_null_counts = {feature: int(output[feature].notna().sum()) for feature in new_features_all}
    count_stats = {
        "min": float(output[COUNT_COL].min()),
        "mean": float(output[COUNT_COL].mean()),
        "max": float(output[COUNT_COL].max()),
    }
    info = {
        "path": str(path),
        "original_rows": original_rows,
        "output_rows": len(output),
        "original_cols": original_cols,
        "cols_after_old_window_removal": len(frame.columns),
        "output_cols": len(output.columns),
        "removed_old_window_features": removed_old_cols,
        "well_counts": well_counts,
        "win_count_stats": count_stats,
    }
    return output, non_null_counts, info


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_non_null_counts(
    features: List[str],
    train_counts: Dict[str, int],
    test_counts: Dict[str, int],
) -> None:
    print("\n新增窗口特征非空数量:")
    for feature in features:
        print(f"  {feature}: train={train_counts.get(feature, 0)}, test={test_counts.get(feature, 0)}")


def print_well_counts(title: str, well_counts: Dict[str, int]) -> None:
    print(f"\n{title} 每口井点数:")
    for well, count in sorted(well_counts.items()):
        print(f"  {well}: {count}")


def main() -> None:
    args = parse_args()
    if args.min_points < 2:
        raise ValueError("--min_points 至少应为 2")
    if args.window_half_width <= 0:
        raise ValueError("--window_half_width 必须大于 0")

    data_dir = resolve_path(args.data_dir)
    train_csv = data_dir / TRAIN_NAME
    test_csv = data_dir / TEST_NAME
    train_backup = data_dir / TRAIN_BACKUP_NAME
    test_backup = data_dir / TEST_BACKUP_NAME

    if not train_csv.exists():
        raise FileNotFoundError(f"找不到训练集: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"找不到测试集: {test_csv}")

    train_backup_created = backup_once(train_csv, train_backup)
    test_backup_created = backup_once(test_csv, test_backup)

    new_features_all, new_features_nosw = build_window_feature_names()

    train_frame = load_csv(train_csv)
    test_frame = load_csv(test_csv)

    train_output, train_non_null, train_info = compute_window_features(
        train_frame,
        train_csv,
        window_half_width=args.window_half_width,
        min_points=args.min_points,
    )
    test_output, test_non_null, test_info = compute_window_features(
        test_frame,
        test_csv,
        window_half_width=args.window_half_width,
        min_points=args.min_points,
    )

    save_csv(train_output, train_csv)
    save_csv(test_output, test_csv)

    stats_path = data_dir / "window_feature_stats.json"
    all_txt_path = data_dir / "window_features_all.txt"
    nosw_txt_path = data_dir / "window_features_nosw.txt"

    stats = {
        "window_half_width_m": args.window_half_width,
        "min_points": args.min_points,
        "well_column": WELL_COL,
        "depth_column": DEPTH_COL,
        "window_base_columns": WINDOW_BASE_COLUMNS,
        "new_window_features_all": new_features_all,
        "new_window_features_nosw": new_features_nosw,
        "train_csv": str(train_csv.relative_to(SCRIPT_DIR)) if train_csv.is_relative_to(SCRIPT_DIR) else str(train_csv),
        "test_csv": str(test_csv.relative_to(SCRIPT_DIR)) if test_csv.is_relative_to(SCRIPT_DIR) else str(test_csv),
        "train_backup": str(train_backup.relative_to(SCRIPT_DIR)) if train_backup.is_relative_to(SCRIPT_DIR) else str(train_backup),
        "test_backup": str(test_backup.relative_to(SCRIPT_DIR)) if test_backup.is_relative_to(SCRIPT_DIR) else str(test_backup),
        "train_backup_created": train_backup_created,
        "test_backup_created": test_backup_created,
        "train_info": train_info,
        "test_info": test_info,
        "train_non_null_counts": train_non_null,
        "test_non_null_counts": test_non_null,
    }
    save_json(stats_path, stats)
    write_lines(all_txt_path, new_features_all)
    write_lines(nosw_txt_path, new_features_nosw)

    print(f"train.csv 路径: {train_csv}")
    print(f"test.csv 路径: {test_csv}")
    print(f"train 备份路径: {train_backup} ({'新建' if train_backup_created else '已存在，未覆盖'})")
    print(f"test 备份路径: {test_backup} ({'新建' if test_backup_created else '已存在，未覆盖'})")
    print(f"train 原始行数 / 输出行数: {train_info['original_rows']} / {train_info['output_rows']}")
    print(f"test 原始行数 / 输出行数: {test_info['original_rows']} / {test_info['output_rows']}")
    print(f"train 原始列数 / 输出列数: {train_info['original_cols']} / {train_info['output_cols']}")
    print(f"test 原始列数 / 输出列数: {test_info['original_cols']} / {test_info['output_cols']}")
    print(f"新增窗口特征数: {len(new_features_all)}")
    print_non_null_counts(new_features_all, train_non_null, test_non_null)
    print_well_counts("train", train_info["well_counts"])
    print_well_counts("test", test_info["well_counts"])
    print(
        "\nwin0p5m_count 统计:"
        f"\n  train: min={train_info['win_count_stats']['min']:.0f}, "
        f"mean={train_info['win_count_stats']['mean']:.2f}, "
        f"max={train_info['win_count_stats']['max']:.0f}"
        f"\n  test: min={test_info['win_count_stats']['min']:.0f}, "
        f"mean={test_info['win_count_stats']['mean']:.2f}, "
        f"max={test_info['win_count_stats']['max']:.0f}"
    )
    print(f"\nwindow_feature_stats.json 保存路径: {stats_path}")
    print(f"window_features_all.txt 保存路径: {all_txt_path}")
    print(f"window_features_nosw.txt 保存路径: {nosw_txt_path}")


if __name__ == "__main__":
    main()

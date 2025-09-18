#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改良版: 複数の results_* ディレクトリを横断集計し、見やすい表と総合評価(推奨手法)を表示

従来:
  - 1つのルート配下 (results_v5/v6 など) のみを探索

本改良:
  - 複数ルート (--roots) を同時に探索・統合できる
  - 各ルートごとの要約に加えて、「全ルート統合（Overall）」の集計を表示
  - 見やすい横断表（Method × Root のピボット風）を出力（Verification Basic の平均）
  - ばらつき（標準偏差）や総合スコアを用いた推奨手法表示（--recommend）
  - オプションで CSV 出力（--csv-out）

対象ログ:
1) 学習時評価ログ: evaluation_v*.log
   - 手法（--- [METHOD] ---）ごとに [Summary] Macro Recall (基本ラベル) / (基本+応用) を抽出
   - ラベル 6A / 6B / 6C の Correct（平均/最小/最大/中央値/合計）と Recall（平均）
     を「各ラベルの再現率（代表ノード群ベース）」から集計

2) 検証時評価ログ: verification_results/*_som/*_verification.log
   - 手法ごとに [Summary] Macro Recall (基本ラベル) / (基本+応用) を抽出

3) 学習結果: learning_result/*_som/*_results.log
   - NodewiseMatchRate（最終値）を集計（Mean/Min/Median/Max と matched/total の合計および全体比）

表示する表では Min / Max の seed も併記（どの seed で出た値か）。

使い方（例）:
  # デフォルトでは src/PressurePattern/ 直下にある以下3つが存在すれば自動探索します:
  #   results_v6_iter100, results_v6_iter1000, results_v6_iter10000
  python src/PressurePattern/search_results_v7.py

  # 任意のディレクトリを複数指定して横断集計
  python src/PressurePattern/search_results_v7.py --roots \
    src/PressurePattern/results_v6_iter100 \
    src/PressurePattern/results_v6_iter1000 \
    src/PressurePattern/results_v6_iter10000

  # 従来どおり単一ルートで集計（後方互換）
  python src/PressurePattern/search_results_v7.py --root src/PressurePattern/results_v6_iter100

  # 推奨手法の算出を表示（検証Basic重視、Combo/安定性/Nodewiseも加味）
  # すべて表示したい場合は --topk 0 を指定
  python src/PressurePattern/search_results_v7.py --recommend --topk 0

  # 横断ピボット表（Verification Basic の平均）の CSV を出力
  python src/PressurePattern/search_results_v7.py --csv-out summary_ver_basic.csv

オプション:
  --roots       複数の results ディレクトリを指定（最優先）
  --root        単一の results ディレクトリを指定（後方互換）
  --sort        並び順（rank/name/basic_combo）: デフォルト rank
  --precision   小数点以下の表示桁数: デフォルト 3
  --recommend   総合スコアに基づく推奨手法 Top-K を表示
  --topk        推奨手法の件数（--recommend 使用時のみ）: 0 以下で全件表示（デフォルト 0）
  --csv-out     横断ピボット表（Verification Basic 平均）の CSV 出力先
"""

import os
import re
import argparse
from typing import Dict, List, Tuple, Any, Optional
import math
import statistics as stats
from decimal import Decimal, ROUND_HALF_UP
import csv


HEADER_RE = re.compile(r'^--- \[(.+?)\] ---')
BASIC_SUMMARY_RE = re.compile(r'^\[Summary\]\s*Macro Recall \(基本ラベル\)\s*=\s*([0-9.]+)')
COMBO_SUMMARY_RE = re.compile(r'^\[Summary\]\s*Macro Recall \(基本\+応用\)\s*=\s*([0-9.]+)')
# NodewiseMatchRate (from *_results.log, e.g., "NodewiseMatchRate = 0.358696 (matched 33/92 nodes)")
NODEWISE_RE = re.compile(r'NodewiseMatchRate\s*=\s*([0-9.]+)\s*\(matched\s*(\d+)\s*/\s*(\d+)\s*nodes\)')

# 「各ラベルの再現率（代表ノード群ベース）」ブロック中の 6A/6B/6C 行を抽出
# 例: " - 6A : N=   7 Correct=   3 Recall=0.4286 代表=[...]"
LABEL_LINE_RE = re.compile(
    r'^-\s*(6[ABC])\s*:\s*N=\s*(\d+)\s+Correct=\s*(\d+)\s+Recall=([0-9.]+)'
)
# verification 用（例: "- 6A : N_base=   0 Correct_base=   0 Recall_base=0.0000 | N_comp=  28 Correct_comp=   0 Recall_comp=0.0000"）
VER_LABEL_LINE_RE = re.compile(
    r'^-\s*(6[ABC])\s*:\s*N_base=\s*(\d+)\s+Correct_base=\s*(\d+)\s+Recall_base=([0-9.]+)\s*\|\s*N_comp=\s*(\d+)\s+Correct_comp=\s*(\d+)\s+Recall_comp=([0-9.]+)'
)
# verification 用（base のみ。例: "- 6A : N_base=   1 Correct_base=   0 Recall_base=0.0000"）
VER_LABEL_BASE_ONLY_RE = re.compile(
    r'^-\s*(6[ABC])\s*:\s*N_base=\s*(\d+)\s+Correct_base=\s*(\d+)\s+Recall_base=([0-9.]+)'
)

LABELS_TARGET = ("6A", "6B", "6C")
# 全15の基本ラベル（列順）
LABELS_ALL = ("1", "2A", "2B", "2C", "2D", "3A", "3B", "3C", "3D", "4A", "4B", "5", "6A", "6B", "6C")

# 学習/評価ログ（results.log/evaluation.log）用: 全ラベル行
ALL_LABEL_LINE_RE = re.compile(
    r'^-\s*(1|2A|2B|2C|2D|3A|3B|3C|3D|4A|4B|5|6A|6B|6C)\s*:\s*N=\s*(\d+)\s+Correct=\s*(\d+)\s+Recall=([0-9.]+)'
)

# 検証ログ用（base | comp の両方が並記されている行）
VER_ALL_LABEL_LINE_RE = re.compile(
    r'^-\s*(1|2A|2B|2C|2D|3A|3B|3C|3D|4A|4B|5|6A|6B|6C)\s*:\s*N_base=\s*(\d+)\s+Correct_base=\s*(\d+)\s+Recall_base=([0-9.]+)\s*\|\s*N_comp=.*$'
)
# 検証ログ用（base のみの行）
VER_ALL_LABEL_BASE_ONLY_RE = re.compile(
    r'^-\s*(1|2A|2B|2C|2D|3A|3B|3C|3D|4A|4B|5|6A|6B|6C)\s*:\s*N_base=\s*(\d+)\s+Correct_base=\s*(\d+)\s+Recall_base=([0-9.]+)'
)


def get_seed_from_path(path: str) -> Optional[int]:
    m = re.search(r'seed(\d+)', path)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def parse_log(log_path: str) -> Dict[str, Dict[str, Any]]:
    """
    1つの evaluation_v*.log をパースして、
    {
      method_name: {
        "basic": float or None,
        "combo": float or None,
        "labels": {
          "6A": {"correct": int, "recall": float},
          "6B": {...},
          "6C": {...}
        }
      }, ...
    } を返す
    """
    methods: Dict[str, Dict[str, Any]] = {}
    current_method: str = ""
    in_basic_label_section: bool = False

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()

                # メソッドヘッダ
                m_header = HEADER_RE.match(line)
                if m_header:
                    current_method = m_header.group(1).strip()
                    if current_method not in methods:
                        methods[current_method] = {
                            "basic": None,
                            "combo": None,
                            "labels": {}
                        }
                    in_basic_label_section = False  # ヘッダを跨いだら一旦解除
                    continue

                if not current_method:
                    # メソッドブロックの外はスキップ
                    continue

                # セクション開始/終了の検知
                if "【各ラベルの再現率（代表ノード群ベース）】" in line:
                    in_basic_label_section = True
                    continue
                if line.startswith("【複合ラベル考慮の再現率（基本+応用）】"):
                    in_basic_label_section = False
                    # ここからは複合側のラベル表になるので 6A/6B/6C の抽出は行わない
                    #（要件は代表ノード群ベースの値を使うため）
                    continue

                # 要約（Summary）
                m_basic = BASIC_SUMMARY_RE.match(line)
                if m_basic:
                    try:
                        methods[current_method]["basic"] = float(m_basic.group(1))
                    except ValueError:
                        pass
                    continue

                m_combo = COMBO_SUMMARY_RE.match(line)
                if m_combo:
                    try:
                        methods[current_method]["combo"] = float(m_combo.group(1))
                    except ValueError:
                        pass
                    continue

                # ラベル 6A/6B/6C の抽出（代表ノード群ベース）
                if in_basic_label_section:
                    m_label = LABEL_LINE_RE.match(line)
                    if m_label:
                        lab = m_label.group(1)
                        if lab in LABELS_TARGET:
                            try:
                                # n = int(m_label.group(2))  # N は今回は未使用
                                correct = int(m_label.group(3))
                                recall = float(m_label.group(4))
                                methods[current_method]["labels"][lab] = {
                                    "correct": correct,
                                    "recall": recall,
                                }
                            except ValueError:
                                pass
    except FileNotFoundError:
        pass

    return methods


def parse_verification_log_for_summaries(log_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    verification_results/*_som/*_verification.log から
    [Summary] Macro Recall (基本ラベル) / (基本+応用) を抽出する
    """
    basic_val: Optional[float] = None
    combo_val: Optional[float] = None
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                m_basic = BASIC_SUMMARY_RE.match(line)
                if m_basic:
                    try:
                        basic_val = float(m_basic.group(1))
                    except Exception:
                        pass
                    continue
                m_combo = COMBO_SUMMARY_RE.match(line)
                if m_combo:
                    try:
                        combo_val = float(m_combo.group(1))
                    except Exception:
                        pass
                    continue
    except FileNotFoundError:
        pass
    return basic_val, combo_val


def collect_logs(root: str) -> Tuple[List[str], List[str], Dict[str, Dict[str, Any]]]:
    """
    root 以下からログを再帰的に収集し、手法別に集約する。
    返り値:
      (eval_log_paths, ver_log_paths, aggregate)
      aggregate: {
        method: {
          "basic": [float, ...],
          "basic_pairs": [(float, seed or None), ...],
          "combo": [float, ...],
          "combo_pairs": [(float, seed or None), ...],

          "ver_basic": [float, ...],
          "ver_basic_pairs": [(float, seed or None), ...],
          "ver_combo": [float, ...],
          "ver_combo_pairs": [(float, seed or None), ...],

          "nodewise": [float, ...],
          "nodewise_pairs": [(float, seed or None), ...],
          "nodewise_matched": [int, ...],
          "nodewise_total": [int, ...],

          "labels": {
            "6A": {"correct_sum": int, "corrects": [int, ...], "recalls": [float, ...], "count": int},
            "6B": {...},
            "6C": {...}
          }
        }
      }
    """
    eval_log_paths: List[str] = []
    ver_log_paths: List[str] = []

    # 収集: evaluation_v*.log
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if re.match(r"evaluation_v\d+\.log$", fn):
                eval_log_paths.append(os.path.join(dirpath, fn))

    aggregate: Dict[str, Dict[str, Any]] = {}

    # 初期化ヘルパ
    def ensure_method(method: str):
        if method not in aggregate:
            aggregate[method] = {
                "basic": [],
                "basic_pairs": [],
                "combo": [],
                "combo_pairs": [],
                "ver_basic": [],
                "ver_basic_pairs": [],
                "ver_combo": [],
                "ver_combo_pairs": [],
                # Node-wise metrics from *_results.log
                "nodewise": [],
                "nodewise_pairs": [],
                "nodewise_matched": [],
                "nodewise_total": [],
                # Typhoon detection metric (Recall-based)
                "typhoon": [],
                "typhoon_pairs": [],
                "typhoon_ver": [],
                "typhoon_ver_pairs": [],
                # 検証時の 6A/6B/6C 統計（従来）
                "ver_labels": {
                    "6A": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6B": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6C": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                },
                # 学習時の 6A/6B/6C 統計（従来）
                "labels": {
                    "6A": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6B": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6C": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                },
                # 追加: 全15基本ラベルの再現率（学習・検証）をラベル別に蓄積
                "train_label_recalls": {lab: [] for lab in LABELS_ALL},
                "ver_label_recalls": {lab: [] for lab in LABELS_ALL},
            }

    # 解析: evaluation_v*.log
    for p in sorted(eval_log_paths):
        parsed = parse_log(p)
        seed = get_seed_from_path(p)
        for method, vals in parsed.items():
            ensure_method(method)
            # Summary
            if vals.get("basic") is not None:
                aggregate[method]["basic"].append(vals["basic"])  # type: ignore[index]
                aggregate[method]["basic_pairs"].append((vals["basic"], seed))  # type: ignore[index]
            if vals.get("combo") is not None:
                aggregate[method]["combo"].append(vals["combo"])  # type: ignore[index]
                aggregate[method]["combo_pairs"].append((vals["combo"], seed))  # type: ignore[index]
            # Labels 6A/6B/6C（代表ノード群ベース）
            labels: Dict[str, Dict[str, float]] = vals.get("labels", {})
            for lab in LABELS_TARGET:
                info = labels.get(lab)
                if info:
                    try:
                        c = int(info["correct"])  # type: ignore[arg-type]
                        r = float(info["recall"])  # type: ignore[arg-type]
                        aggregate[method]["labels"][lab]["correct_sum"] += c  # type: ignore[index]
                        aggregate[method]["labels"][lab]["corrects"].append(c)  # type: ignore[index]
                        aggregate[method]["labels"][lab]["recalls"].append(r)  # type: ignore[index]
                        aggregate[method]["labels"][lab]["count"] += 1  # type: ignore[index]
                    except Exception:
                        # パース失敗時はスキップ
                        pass
            # 台風補足（Recall）: 6A/6B の Recall を利用可能ラベルで平均して seed 指標とする
            try:
                ty_vals: List[float] = []
                info_6a = labels.get("6A")
                info_6b = labels.get("6B")
                if info_6a is not None and isinstance(info_6a.get("recall"), (int, float)):
                    ty_vals.append(float(info_6a["recall"]))
                if info_6b is not None and isinstance(info_6b.get("recall"), (int, float)):
                    ty_vals.append(float(info_6b["recall"]))
                if ty_vals:
                    ty_metric = sum(ty_vals) / len(ty_vals)
                    aggregate[method]["typhoon"].append(ty_metric)  # type: ignore[index]
                    aggregate[method]["typhoon_pairs"].append((ty_metric, seed))  # type: ignore[index]
            except Exception:
                pass

    # 追加収集: learning_result/*_som/*_results.log から NodewiseMatchRate（最終値）を集計
    results_logs: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith("_results.log"):
                results_logs.append(os.path.join(dirpath, fn))

    for rp in sorted(results_logs):
        # 推定手法名: ディレクトリ名 '<name>_som' の前半を大文字化
        method_dir = os.path.basename(os.path.dirname(rp))  # e.g., 'euclidean_som'
        base_name = method_dir.rsplit("_som", 1)[0].upper()
        ensure_method(base_name)
        last_tuple = None  # (rate, matched, total)
        try:
            with open(rp, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    m = NODEWISE_RE.search(raw)
                    if m:
                        try:
                            rate = float(m.group(1))
                            matched = int(m.group(2))
                            total = int(m.group(3))
                            last_tuple = (rate, matched, total)
                        except Exception:
                            pass
        except FileNotFoundError:
            last_tuple = None

        if last_tuple:
            rate, matched, total = last_tuple
            aggregate[base_name]["nodewise"].append(rate)  # type: ignore[index]
            aggregate[base_name]["nodewise_matched"].append(matched)  # type: ignore[index]
            aggregate[base_name]["nodewise_total"].append(total)  # type: ignore[index]
            aggregate[base_name]["nodewise_pairs"].append((rate, get_seed_from_path(rp)))  # type: ignore[index]
        # 追加収集: *_results.log 終盤の「各ラベルの再現率（代表ノード群ベース）」から全15基本ラベルの Recall を学習側として集計
        try:
            in_basic = False
            with open(rp, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    s = raw.strip()
                    if "【各ラベルの再現率（代表ノード群ベース）】" in s:
                        in_basic = True
                        continue
                    if s.startswith("[Summary]"):
                        in_basic = False
                    if in_basic:
                        m_all = ALL_LABEL_LINE_RE.match(s)
                        if m_all:
                            lab = m_all.group(1)
                            try:
                                rec = float(m_all.group(4))
                            except Exception:
                                continue
                            if lab in LABELS_ALL:
                                aggregate[base_name]["train_label_recalls"][lab].append(rec)  # type: ignore[index]
        except Exception:
            pass

    # 収集/解析: verification_results/*_som/*_verification.log
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith("_verification.log"):
                vp = os.path.join(dirpath, fn)
                ver_log_paths.append(vp)
                method_dir = os.path.basename(os.path.dirname(vp))  # '<name>_som'
                base_name = method_dir.rsplit("_som", 1)[0].upper()
                ensure_method(base_name)
                vb, vc = parse_verification_log_for_summaries(vp)
                seed = get_seed_from_path(vp)
                if vb is not None:
                    aggregate[base_name]["ver_basic"].append(vb)  # type: ignore[index]
                    aggregate[base_name]["ver_basic_pairs"].append((vb, seed))  # type: ignore[index]
                if vc is not None:
                    aggregate[base_name]["ver_combo"].append(vc)  # type: ignore[index]
                    aggregate[base_name]["ver_combo_pairs"].append((vc, seed))  # type: ignore[index]
                # Typhoon (verification) 6A/6B recall per seed -> average, and collect verification label stats (6A/6B/6C)
                try:
                    ty_vals_ver: List[float] = []
                    with open(vp, "r", encoding="utf-8", errors="ignore") as vf:
                        for raw_v in vf:
                            s = raw_v.strip()
                            # まず verification 専用フォーマットを試す（全ラベル→6A/6B/6Cの順）
                            m_all = VER_ALL_LABEL_LINE_RE.match(s)
                            if m_all:
                                lab = m_all.group(1)
                                try:
                                    corr_base = int(m_all.group(3))
                                    rec_base = float(m_all.group(4))
                                except Exception:
                                    continue
                                if lab in ("6A", "6B"):
                                    ty_vals_ver.append(rec_base)
                                if lab in LABELS_TARGET:
                                    try:
                                        aggregate[base_name]["ver_labels"][lab]["correct_sum"] += corr_base  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["corrects"].append(corr_base)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["recalls"].append(rec_base)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["count"] += 1  # type: ignore[index]
                                    except Exception:
                                        pass
                                if lab in LABELS_ALL:
                                    try:
                                        aggregate[base_name]["ver_label_recalls"][lab].append(rec_base)  # type: ignore[index]
                                    except Exception:
                                        pass
                                continue
                            m_all_base = VER_ALL_LABEL_BASE_ONLY_RE.match(s)
                            if m_all_base:
                                lab = m_all_base.group(1)
                                try:
                                    corr_base = int(m_all_base.group(3))
                                    rec_base = float(m_all_base.group(4))
                                except Exception:
                                    continue
                                if lab in ("6A", "6B"):
                                    ty_vals_ver.append(rec_base)
                                if lab in LABELS_TARGET:
                                    try:
                                        aggregate[base_name]["ver_labels"][lab]["correct_sum"] += corr_base  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["corrects"].append(corr_base)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["recalls"].append(rec_base)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["count"] += 1  # type: ignore[index]
                                    except Exception:
                                        pass
                                if lab in LABELS_ALL:
                                    try:
                                        aggregate[base_name]["ver_label_recalls"][lab].append(rec_base)  # type: ignore[index]
                                    except Exception:
                                        pass
                                continue
                            m_ver = VER_LABEL_LINE_RE.match(s)
                            if m_ver:
                                lab = m_ver.group(1)
                                try:
                                    corr_base = int(m_ver.group(3))
                                    rec_base = float(m_ver.group(4))
                                except Exception:
                                    continue
                                if lab in ("6A", "6B"):
                                    ty_vals_ver.append(rec_base)
                                if lab in LABELS_TARGET:
                                    try:
                                        aggregate[base_name]["ver_labels"][lab]["correct_sum"] += corr_base  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["corrects"].append(corr_base)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["recalls"].append(rec_base)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["count"] += 1  # type: ignore[index]
                                    except Exception:
                                        pass
                                # 追加: 全15基本ラベルの base Recall を収集
                                if lab in LABELS_ALL:
                                    try:
                                        aggregate[base_name]["ver_label_recalls"][lab].append(rec_base)  # type: ignore[index]
                                    except Exception:
                                        pass
                                continue
                            # 追加: 複合系(Comp)の無い base のみの行にも対応
                            m_ver_base = VER_LABEL_BASE_ONLY_RE.match(s)
                            if m_ver_base:
                                lab = m_ver_base.group(1)
                                try:
                                    corr_base = int(m_ver_base.group(3))
                                    rec_base = float(m_ver_base.group(4))
                                except Exception:
                                    continue
                                if lab in ("6A", "6B"):
                                    ty_vals_ver.append(rec_base)
                                if lab in LABELS_TARGET:
                                    try:
                                        aggregate[base_name]["ver_labels"][lab]["correct_sum"] += corr_base
                                        aggregate[base_name]["ver_labels"][lab]["corrects"].append(corr_base)
                                        aggregate[base_name]["ver_labels"][lab]["recalls"].append(rec_base)
                                        aggregate[base_name]["ver_labels"][lab]["count"] += 1
                                    except Exception:
                                        pass
                                # 追加: 全15基本ラベルの base Recall を収集
                                if lab in LABELS_ALL:
                                    try:
                                        aggregate[base_name]["ver_label_recalls"][lab].append(rec_base)  # type: ignore[index]
                                    except Exception:
                                        pass
                                continue
                            # フォールバック: 学習形式に近い行にも対応
                            m_lab = LABEL_LINE_RE.match(s)
                            if m_lab:
                                lab = m_lab.group(1)
                                try:
                                    corr_v = int(m_lab.group(3))
                                    rec_v = float(m_lab.group(4))
                                except Exception:
                                    continue
                                if lab in ("6A", "6B"):
                                    ty_vals_ver.append(rec_v)
                                if lab in LABELS_TARGET:
                                    try:
                                        aggregate[base_name]["ver_labels"][lab]["correct_sum"] += corr_v  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["corrects"].append(corr_v)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["recalls"].append(rec_v)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["count"] += 1  # type: ignore[index]
                                    except Exception:
                                        pass
                                # 追加: 旧式（学習形式）行でも base として扱い収集
                                if lab in LABELS_ALL:
                                    try:
                                        aggregate[base_name]["ver_label_recalls"][lab].append(rec_v)  # type: ignore[index]
                                    except Exception:
                                        pass
                    if ty_vals_ver:
                        ty_metric_ver = sum(ty_vals_ver) / len(ty_vals_ver)
                        aggregate[base_name]["typhoon_ver"].append(ty_metric_ver)  # type: ignore[index]
                        aggregate[base_name]["typhoon_ver_pairs"].append((ty_metric_ver, seed))  # type: ignore[index]
                except Exception:
                    pass

    return eval_log_paths, ver_log_paths, aggregate


def mean_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def min_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return min(values)


def max_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return max(values)


def median_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return stats.median(values)


def std_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    try:
        return stats.stdev(values)
    except Exception:
        return float("nan")


def has_any_values(aggregate: Dict[str, Dict[str, Any]], key: str) -> bool:
    """aggregate の各 method で指定 key の配列に1つでも値があれば True。"""
    for _m, metrics in aggregate.items():
        vals = metrics.get(key, [])
        if isinstance(vals, list) and len(vals) > 0:
            return True
    return False


def fmt_float(v: float, prec: int) -> str:
    if math.isnan(v):
        return "NaN"
    try:
        d = Decimal(str(v))
        exp = Decimal('1').scaleb(-prec)  # e.g., prec=2 -> 0.01
        rounded = d.quantize(exp, rounding=ROUND_HALF_UP)
        return str(rounded)  # keeps trailing zeros (e.g., '1.20')
    except Exception:
        # Fallback to standard formatting
        return f"{v:.{prec}f}"


def fmt_seed(seed: Optional[int]) -> str:
    return f"{seed:d}" if seed is not None else "-"


def find_extreme_seeds(pairs: List[Tuple[float, Optional[int]]]) -> Tuple[Optional[int], Optional[int], float, float]:
    """
    pairs: [(value, seed), ...]
    return: (min_seed, max_seed, min_val, max_val)
    """
    if not pairs:
        return None, None, float("nan"), float("nan")
    values = [v for v, _ in pairs]
    min_v = min(values)
    max_v = max(values)
    min_seed = next((s for v, s in pairs if v == min_v), None)
    max_seed = next((s for v, s in pairs if v == max_v), None)
    return min_seed, max_seed, min_v, max_v


def seed_means_from_pairs(pairs: List[Tuple[float, Optional[int]]]) -> Dict[int, float]:
    """同一 seed が複数回出現する場合は平均して seed->平均値 の辞書を返す。"""
    from collections import defaultdict
    d: Dict[int, List[float]] = defaultdict(list)
    for v, s in pairs:
        if s is None:
            continue
        try:
            d[int(s)].append(float(v))
        except Exception:
            continue
    return {s: (sum(vals) / len(vals)) for s, vals in d.items() if len(vals) > 0}


def select_best_seed_balanced(maps: List[Dict[int, float]]) -> Optional[int]:
    """
    5 指標（ver_basic, ver_combo, basic, combo, nodewise）の seed 別値を受け取り、
    - 出現指標数が最大の seed を優先（全5指標に揃っていれば最優先）
    - 次点として平均値（利用可能な指標の単純平均）が最大の seed を選ぶ
    """
    from collections import Counter
    if not maps:
        return None
    counts = Counter()
    for m in maps:
        counts.update(m.keys())
    if not counts:
        return None

    max_cov = max(counts.values())
    while max_cov >= 1:
        candidates = [s for s, c in counts.items() if c == max_cov]
        if candidates:
            best_seed: Optional[int] = None
            best_avg = -float("inf")
            for s in candidates:
                vals = [m[s] for m in maps if s in m]
                if not vals:
                    continue
                avg = sum(vals) / len(vals)
                if avg > best_avg or (avg == best_avg and (best_seed is None or s < best_seed)):
                    best_seed = s
                    best_avg = avg
            if best_seed is not None:
                return best_seed
        max_cov -= 1
    return None


def pair_means_by_root_seed(pairs: List[Tuple[float, Optional[int]]], root_label: str) -> Dict[Tuple[str, int], float]:
    """
    同一 (root, seed) が複数回出現する場合は平均して {(root, seed): 平均値} を返す。
    """
    base = seed_means_from_pairs(pairs)
    return {(root_label, s): v for s, v in base.items()}


def select_best_root_seed_balanced(maps: List[Dict[Tuple[str, int], float]]) -> Optional[Tuple[str, int]]:
    """
    5 指標の (root, seed) 別値を受け取り、
    - 出現指標数が最大の (root, seed) を優先
    - 次に平均値（利用可能な指標の単純平均）が最大の (root, seed)
    """
    from collections import Counter
    if not maps:
        return None
    counts = Counter()
    for m in maps:
        counts.update(m.keys())
    if not counts:
        return None
    max_cov = max(counts.values())
    while max_cov >= 1:
        candidates = [k for k, c in counts.items() if c == max_cov]
        if candidates:
            best_key: Optional[Tuple[str, int]] = None
            best_avg = -float("inf")
            for key in candidates:
                vals = [m[key] for m in maps if key in m]
                if not vals:
                    continue
                avg = sum(vals) / len(vals)
                if avg > best_avg or (avg == best_avg and (best_key is None or key < best_key)):
                    best_key = key
                    best_avg = avg
            if best_key is not None:
                return best_key
        max_cov -= 1
    return None

def ensure_method_in_agg(agg: Dict[str, Dict[str, Any]], method: str):
    if method not in agg:
        agg[method] = {
            "basic": [],
            "basic_pairs": [],
            "combo": [],
            "combo_pairs": [],
            "ver_basic": [],
            "ver_basic_pairs": [],
            "ver_combo": [],
            "ver_combo_pairs": [],
            "nodewise": [],
            "nodewise_pairs": [],
            "nodewise_matched": [],
            "nodewise_total": [],
            "typhoon": [],
            "typhoon_pairs": [],
            "typhoon_ver": [],
            "typhoon_ver_pairs": [],
            "ver_labels": {
                "6A": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                "6B": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                "6C": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
            },
            "labels": {
                "6A": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                "6B": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                "6C": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
            },
            "train_label_recalls": {lab: [] for lab in LABELS_ALL},
            "ver_label_recalls": {lab: [] for lab in LABELS_ALL},
        }


def merge_aggregates(aggregates: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    複数の aggregate を統合
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for agg in aggregates:
        for method, m in agg.items():
            ensure_method_in_agg(merged, method)
            # リスト系を伸長
            for key in [
                "basic", "basic_pairs",
                "combo", "combo_pairs",
                "ver_basic", "ver_basic_pairs",
                "ver_combo", "ver_combo_pairs",
                "nodewise", "nodewise_pairs",
                "nodewise_matched", "nodewise_total",
                "typhoon", "typhoon_pairs",
                "typhoon_ver", "typhoon_ver_pairs",
            ]:
                merged[method][key].extend(m.get(key, []))  # type: ignore[index]
            # ラベル系をマージ
            for lab in LABELS_TARGET:
                dst = merged[method]["labels"][lab]  # type: ignore[index]
                src = m.get("labels", {}).get(lab, {})
                dst["correct_sum"] += src.get("correct_sum", 0)  # type: ignore[index]
                dst["count"] += src.get("count", 0)  # type: ignore[index]
                dst["corrects"].extend(src.get("corrects", []))  # type: ignore[index]
                dst["recalls"].extend(src.get("recalls", []))  # type: ignore[index]
            # Merge verification label stats as well
            for lab in LABELS_TARGET:
                dstv = merged[method].setdefault("ver_labels", {
                    "6A": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6B": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6C": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                })[lab]  # type: ignore[index]
                srcv = m.get("ver_labels", {}).get(lab, {})
                dstv["correct_sum"] += srcv.get("correct_sum", 0)  # type: ignore[index]
                dstv["count"] += srcv.get("count", 0)  # type: ignore[index]
                dstv["corrects"].extend(srcv.get("corrects", []))  # type: ignore[index]
                dstv["recalls"].extend(srcv.get("recalls", []))  # type: ignore[index]
            # 追加: 全15基本ラベル（学習/検証）の再現率をマージ
            dst_train = merged[method].setdefault("train_label_recalls", {lab: [] for lab in LABELS_ALL})
            src_train = m.get("train_label_recalls", {})
            for lab in LABELS_ALL:
                dst_train.setdefault(lab, [])
                dst_train[lab].extend(src_train.get(lab, []))
            dst_ver = merged[method].setdefault("ver_label_recalls", {lab: [] for lab in LABELS_ALL})
            src_ver = m.get("ver_label_recalls", {})
            for lab in LABELS_ALL:
                dst_ver.setdefault(lab, [])
                dst_ver[lab].extend(src_ver.get(lab, []))
    return merged


def print_table(aggregate: Dict[str, Dict[str, Any]], title: str, value_key: str, pair_key: str, sort_mode: str, prec: int):
    header = (
        f'{title:24s} '
        f'{"N":>5s} {"Mean":>10s} {"Min":>10s} {"Median":>10s} {"Max":>10s} '
        f'{"MinSeed":>8s} {"MaxSeed":>8s}'
    )
    print(header)
    print("-" * len(header))
    # 並び順関数
    def key_by_name(item):
        return item[0]

    def key_by_mean_key(item):
        name, metrics = item
        return -mean_or_nan(metrics.get(value_key, []))  # type: ignore[arg-type,index]

    items = list(aggregate.items())
    if sort_mode == "name":
        items.sort(key=key_by_name)
    else:
        items.sort(key=key_by_mean_key)

    for method, metrics in items:
        vals: List[float] = metrics.get(value_key, [])  # type: ignore[assignment]
        pairs: List[Tuple[float, Optional[int]]] = metrics.get(pair_key, [])  # type: ignore[assignment]
        n = len(vals)
        mean_v = mean_or_nan(vals)
        min_v = min_or_nan(vals)
        med_v = median_or_nan(vals)
        max_v = max_or_nan(vals)
        min_seed, max_seed, _mv, _xv = find_extreme_seeds(pairs)
        print(
            f"{method:24s} {n:5d} "
            f"{fmt_float(mean_v, prec):>10s} {fmt_float(min_v, prec):>10s} "
            f"{fmt_float(med_v, prec):>10s} {fmt_float(max_v, prec):>10s} "
            f"{fmt_seed(min_seed):>8s} {fmt_seed(max_seed):>8s}"
        )
    print("")


def print_nodewise_table(aggregate: Dict[str, Dict[str, Any]], sort_mode: str, prec: int):
    header_nodewise = (
        f'{"[Nodewise] Method":24s} '
        f'{"N":>5s} {"Mean":>10s} {"Min":>10s} {"Median":>10s} {"Max":>10s} '
        f'{"Σmatch":>10s} {"Σnodes":>10s} {"Overall":>10s} '
        f'{"MinSeed":>8s} {"MaxSeed":>8s}'
    )
    print("==== 手法別 NodewiseMatchRate 統計（learning_result/*_results.log の[Final Metrics]より） ====")
    print(header_nodewise)
    print("-" * len(header_nodewise))

    def key_by_name(item):
        return item[0]

    def key_by_mean_nodewise(item):
        name, metrics = item
        return -mean_or_nan(metrics.get("nodewise", []))  # type: ignore[arg-type,index]

    items3 = list(aggregate.items())
    if sort_mode == "name":
        items3.sort(key=key_by_name)
    else:
        items3.sort(key=key_by_mean_nodewise)

    for method, metrics in items3:
        rates: List[float] = metrics.get("nodewise", [])  # type: ignore[assignment]
        n = len(rates)
        mean_v = mean_or_nan(rates)
        min_v = min_or_nan(rates)
        med_v = median_or_nan(rates)
        max_v = max_or_nan(rates)
        sum_match = sum(metrics.get("nodewise_matched", []))  # type: ignore[arg-type]
        sum_total = sum(metrics.get("nodewise_total", []))  # type: ignore[arg-type]
        overall = (sum_match / sum_total) if sum_total > 0 else float("nan")
        min_seed, max_seed, _mv, _xv = find_extreme_seeds(metrics.get("nodewise_pairs", []))  # type: ignore[arg-type]
        print(
            f"{method:24s} {n:5d} "
            f"{fmt_float(mean_v, prec):>10s} {fmt_float(min_v, prec):>10s} "
            f"{fmt_float(med_v, prec):>10s} {fmt_float(max_v, prec):>10s} "
            f"{sum_match:10d} {sum_total:10d} {fmt_float(overall, prec):>10s} "
            f"{fmt_seed(min_seed):>8s} {fmt_seed(max_seed):>8s}"
        )
    print("")


def print_label_stats_tables(aggregate: Dict[str, Dict[str, Any]], sort_mode: str, prec: int):
    for label_key in LABELS_TARGET:
        title = f"==== ラベル {label_key} の統計（代表ノード群ベース: Correctの平均/最小/最大/中央値/合計、Recallの平均） ===="
        print(title)
        header = (
            f'{"Method":24s} '
            f'{"N":>5s} '
            f'{"Mean_C":>10s} {"Min_C":>10s} {"Med_C":>10s} {"Max_C":>10s} {"Sum_C":>10s} '
            f'{"Mean_R":>10s}'
        )
        print(header)
        print("-" * len(header))
        rows = []
        for method, metrics in aggregate.items():
            info = metrics["labels"][label_key]  # type: ignore[index]
            corrects: List[int] = info["corrects"]  # type: ignore[index]
            recalls: List[float] = info["recalls"]  # type: ignore[index]
            n = len(corrects)
            mean_c = stats.mean(corrects) if n > 0 else float("nan")
            min_c = min(corrects) if n > 0 else float("nan")
            med_c = stats.median(corrects) if n > 0 else float("nan")
            max_c = max(corrects) if n > 0 else float("nan")
            sum_c = info["correct_sum"]  # type: ignore[index]
            mean_r = mean_or_nan(recalls)
            rows.append((method, n, mean_c, min_c, med_c, max_c, sum_c, mean_r))

        if sort_mode == "name":
            rows.sort(key=lambda x: x[0])
        else:
            rows.sort(key=lambda x: (- (x[7] if not math.isnan(x[7]) else -1.0), x[0]))

        for method, n, mean_c, min_c, med_c, max_c, sum_c, mean_r in rows:
            print(
                f"{method:24s} {n:5d} "
                f"{fmt_float(float(mean_c), prec):>10s} {fmt_float(float(min_c), prec):>10s} "
                f"{fmt_float(float(med_c), prec):>10s} {fmt_float(float(max_c), prec):>10s} "
                f"{sum_c:10d} {fmt_float(mean_r, prec):>10s}"
            )
        print("")
    
    
def print_ver_label_stats_tables(aggregate: Dict[str, Dict[str, Any]], sort_mode: str, prec: int):
    for label_key in LABELS_TARGET:
        title = f"==== 検証 ラベル {label_key} の統計（代表ノード群ベース: Correctの平均/最小/最大/中央値/合計、Recallの平均） ===="
        print(title)
        header = (
            f'{"Method":24s} '
            f'{"N":>5s} '
            f'{"Mean_C":>10s} {"Min_C":>10s} {"Med_C":>10s} {"Max_C":>10s} {"Sum_C":>10s} '
            f'{"Mean_R":>10s}'
        )
        print(header)
        print("-" * len(header))
        rows = []
        for method, metrics in aggregate.items():
            info = metrics.get("ver_labels", {}).get(label_key, {"corrects": [], "recalls": [], "count": 0, "correct_sum": 0})
            corrects: List[int] = info.get("corrects", [])
            recalls: List[float] = info.get("recalls", [])
            n = len(corrects)
            mean_c = stats.mean(corrects) if n > 0 else float("nan")
            min_c = min(corrects) if n > 0 else float("nan")
            med_c = stats.median(corrects) if n > 0 else float("nan")
            max_c = max(corrects) if n > 0 else float("nan")
            sum_c = info.get("correct_sum", 0)
            mean_r = mean_or_nan(recalls)
            rows.append((method, n, mean_c, min_c, med_c, max_c, sum_c, mean_r))
        if sort_mode == "name":
            rows.sort(key=lambda x: x[0])
        else:
            rows.sort(key=lambda x: (- (x[7] if not math.isnan(x[7]) else -1.0), x[0]))
        for method, n, mean_c, min_c, med_c, max_c, sum_c, mean_r in rows:
            print(
                f"{method:24s} {n:5d} "
                f"{fmt_float(float(mean_c), prec):>10s} {fmt_float(float(min_c), prec):>10s} "
                f"{fmt_float(float(med_c), prec):>10s} {fmt_float(float(max_c), prec):>10s} "
                f"{sum_c:10d} {fmt_float(mean_r, prec):>10s}"
            )
        print("")
    
    
def print_cross_root_ver_basic(per_root_aggregates: Dict[str, Dict[str, Dict[str, Any]]], overall_agg: Dict[str, Dict[str, Any]], prec: int):
    """
    横断ピボット風の表: Method × Root で Verification Basic の平均値を表示し、右端に Overall(mean) と Std を付与
    """
    if not per_root_aggregates:
        return
    roots = list(per_root_aggregates.keys())
    roots.sort()

    # ヘッダ
    header = f'{"[Pivot] VerBasic Mean":24s} '
    for r in roots:
        header += f'{os.path.basename(r):>12s} '
    header += f'{"Overall":>12s} {"Std":>12s} {"N_all":>6s}'
    print("==== 横断ピボット表（Verification Basic の平均） ====")
    print(header)
    print("-" * len(header))

    methods = set()
    for agg in per_root_aggregates.values():
        methods |= set(agg.keys())
    methods = sorted(methods)

    # 並びは overall の mean 降順
    def overall_mean(method: str) -> float:
        return mean_or_nan(overall_agg.get(method, {}).get("ver_basic", []))  # type: ignore[arg-type]

    methods.sort(key=lambda m: - (overall_mean(m) if not math.isnan(overall_mean(m)) else -1.0))

    for method in methods:
        row = f"{method:24s} "
        vb_all = overall_agg.get(method, {}).get("ver_basic", [])  # type: ignore[arg-type]
        overall_mean_v = mean_or_nan(vb_all)
        overall_std_v = std_or_nan(vb_all)
        n_all = len(vb_all)
        for r in roots:
            agg = per_root_aggregates[r]
            vals = agg.get(method, {}).get("ver_basic", [])  # type: ignore[arg-type]
            row += f"{fmt_float(mean_or_nan(vals), prec):>12s} "
        row += f"{fmt_float(overall_mean_v, prec):>12s} {fmt_float(overall_std_v, prec):>12s} {n_all:6d}"
        print(row)
    print("")


def print_per_label_pivot(aggregate: Dict[str, Dict[str, Any]], label_source: str, prec: int):
    """
    行=手法, 列=全15基本ラベル で、各ラベルの平均Recall（seed横断平均）を表示するピボット表。
    label_source: "train" -> *_results.log（学習）由来, "ver" -> *_verification.log（検証）由来
    """
    if not aggregate:
        return
    key = "train_label_recalls" if label_source == "train" else "ver_label_recalls"
    title = "横断ピボット表（学習: 基本ラベルごとの平均再現率）" if label_source == "train" else "横断ピボット表（検証: 基本ラベルごとの平均再現率）"
    print(f"==== {title} ====")
    header = f'{"[Pivot] Method":24s} ' + " ".join(f"{lab:>6s}" for lab in LABELS_ALL) + f' {"Overall":>8s} {"N_lab":>6s}'
    print(header)
    print("-" * len(header))

    methods = sorted(aggregate.keys())

    def overall_mean_method(meth: str) -> float:
        recs_map: Dict[str, List[float]] = aggregate.get(meth, {}).get(key, {})  # type: ignore[assignment]
        vals: List[float] = []
        for lab in LABELS_ALL:
            lv = mean_or_nan(recs_map.get(lab, []))
            if not math.isnan(lv):
                vals.append(lv)
        return mean_or_nan(vals) if vals else float("nan")

    methods.sort(key=lambda m: - (overall_mean_method(m) if not math.isnan(overall_mean_method(m)) else -1.0))

    for meth in methods:
        recs_map: Dict[str, List[float]] = aggregate.get(meth, {}).get(key, {})  # type: ignore[assignment]
        label_means: List[float] = []
        cells: List[str] = []
        n_lab = 0
        for lab in LABELS_ALL:
            mv = mean_or_nan(recs_map.get(lab, []))
            cells.append(fmt_float(mv, prec))
            if not math.isnan(mv):
                label_means.append(mv)
                n_lab += 1
        overall_m = mean_or_nan(label_means) if label_means else float("nan")
        print(f"{meth:24s} " + " ".join(f"{c:>6s}" for c in cells) + f" {fmt_float(overall_m, prec):>8s} {n_lab:6d}")
    print("")


def print_all_tables_for_aggregate(aggregate: Dict[str, Dict[str, Any]], context_name: str, sort_mode: str, prec: int):
    print(f"==== 手法別 Macro Recall 統計（学習: 基本ラベル, evaluation_v*.log）[{context_name}] ====")
    print_table(aggregate, "[基本] Method", "basic", "basic_pairs", sort_mode, prec)

    if has_any_values(aggregate, "combo"):
        print(f"==== 手法別 Macro Recall 統計（学習: 基本+応用, evaluation_v*.log）[{context_name}] ====")
        print_table(aggregate, "[基本+応用] Method", "combo", "combo_pairs", sort_mode, prec)

    print(f"==== 手法別 Macro Recall 統計（検証: 基本ラベル, *_verification.log）[{context_name}] ====")
    print_table(aggregate, "[Ver基本] Method", "ver_basic", "ver_basic_pairs", sort_mode, prec)

    if has_any_values(aggregate, "ver_combo"):
        print(f"==== 手法別 Macro Recall 統計（検証: 基本+応用, *_verification.log）[{context_name}] ====")
        print_table(aggregate, "[Ver基本+応用] Method", "ver_combo", "ver_combo_pairs", sort_mode, prec)

    print_nodewise_table(aggregate, sort_mode, prec)
    print_label_stats_tables(aggregate, sort_mode, prec)
    print_ver_label_stats_tables(aggregate, sort_mode, prec)
    # 新規ピボット（全15基本ラベル）
    print_per_label_pivot(aggregate, "train", prec)
    print_per_label_pivot(aggregate, "ver", prec)


def print_overall_tables(overall_agg: Dict[str, Dict[str, Any]], sort_mode: str, prec: int):
    print("==== 手法別 Macro Recall 統計（学習: 基本ラベル, evaluation_v*.log）[Overall] ====")
    print_table(overall_agg, "[基本] Method", "basic", "basic_pairs", sort_mode, prec)

    if has_any_values(overall_agg, "combo"):
        print("==== 手法別 Macro Recall 統計（学習: 基本+応用, evaluation_v*.log）[Overall] ====")
        print_table(overall_agg, "[基本+応用] Method", "combo", "combo_pairs", sort_mode, prec)

    print("==== 手法別 Macro Recall 統計（検証: 基本ラベル, *_verification.log）[Overall] ====")
    print_table(overall_agg, "[Ver基本] Method", "ver_basic", "ver_basic_pairs", sort_mode, prec)

    if has_any_values(overall_agg, "ver_combo"):
        print("==== 手法別 Macro Recall 統計（検証: 基本+応用, *_verification.log）[Overall] ====")
        print_table(overall_agg, "[Ver基本+応用] Method", "ver_combo", "ver_combo_pairs", sort_mode, prec)

    print_nodewise_table(overall_agg, sort_mode, prec)
    print_label_stats_tables(overall_agg, sort_mode, prec)
    print_ver_label_stats_tables(overall_agg, sort_mode, prec)
    # 新規ピボット（全15基本ラベル）
    print_per_label_pivot(overall_agg, "train", prec)
    print_per_label_pivot(overall_agg, "ver", prec)


def recommend_methods(overall_agg: Dict[str, Dict[str, Any]], topk: int, prec: int, context_name: str = "Overall", roots_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None):
    """
    推奨手法を総合スコアで提示。
    スコア定義（Mean と Std を別々に集計; 単純和）:
      Score(SumMean) = BasicMean + ComboMean + TyphoonMean
        - BasicMean  = mean({VerBasicMean,  TrainBasicMean} 利用可能のみ)
        - ComboMean  = mean({VerComboMean,  TrainComboMean} 利用可能のみ)
        - TyphoonMean= TrainTyphoonRecallMean + VerTyphoonRecallMean
      Score(SumStd)  = BasicStd  + ComboStd  + TyphoonStd
        - BasicStd   = mean({VerBasicStd,   TrainBasicStd} 利用可能のみ)
        - ComboStd   = mean({VerComboStd,   TrainComboStd} 利用可能のみ)
        - TyphoonStd = TrainTyphoonRecallStd + VerTyphoonRecallStd
    注: Nodewise は参考指標でありスコアには含めません。
    """
    rows = []
    for method, metrics in overall_agg.items():
        vb_list: List[float] = metrics.get("ver_basic", [])  # type: ignore[assignment]
        vc_list: List[float] = metrics.get("ver_combo", [])  # type: ignore[assignment]
        tb_list: List[float] = metrics.get("basic", [])  # 学習(基本)
        tc_list: List[float] = metrics.get("combo", [])  # 学習(基本+応用)
        vb_mean = mean_or_nan(vb_list)
        vb_std = std_or_nan(vb_list)
        vc_mean = mean_or_nan(vc_list)
        vc_std = std_or_nan(vc_list)
        tb_mean = mean_or_nan(tb_list)
        tb_std = std_or_nan(tb_list)
        tc_mean = mean_or_nan(tc_list)
        tc_std = std_or_nan(tc_list)
        # Typhoon (train/evaluation) recall per seed across 6A/6B
        ty_list: List[float] = metrics.get("typhoon", [])  # type: ignore[assignment]
        ty_mean = mean_or_nan(ty_list)
        ty_std = std_or_nan(ty_list)
        # Typhoon (verification)
        ty_ver_list: List[float] = metrics.get("typhoon_ver", [])  # type: ignore[assignment]
        ty_ver_mean = mean_or_nan(ty_ver_list)
        ty_ver_std = std_or_nan(ty_ver_list)
        # overall nodewise ratio（集計比は参考算出だがスコアには含めない）
        sum_match = sum(metrics.get("nodewise_matched", []))  # type: ignore[arg-type]
        sum_total = sum(metrics.get("nodewise_total", []))  # type: ignore[arg-type]
        nodewise_overall = (sum_match / sum_total) if sum_total > 0 else float("nan")
        # nodewise mean/std
        nw_mean = mean_or_nan(metrics.get("nodewise", []))  # type: ignore[arg-type]
        nw_std = std_or_nan(metrics.get("nodewise", []))  # type: ignore[arg-type]
        # NaN 保護
        vb_m = 0.0 if math.isnan(vb_mean) else vb_mean
        vb_s = 0.0 if math.isnan(vb_std) else vb_std
        vc_m = 0.0 if math.isnan(vc_mean) else vc_mean
        vc_s = 0.0 if math.isnan(vc_std) else vc_std
        tb_m = 0.0 if math.isnan(tb_mean) else tb_mean
        tb_s = 0.0 if math.isnan(tb_std) else tb_std
        tc_m = 0.0 if math.isnan(tc_mean) else tc_mean
        tc_s = 0.0 if math.isnan(tc_std) else tc_std
        nw_m = 0.0 if math.isnan(nw_mean) else nw_mean
        nw_s = 0.0 if math.isnan(nw_std) else nw_std
        ty_m = 0.0 if math.isnan(ty_mean) else ty_mean
        ty_s = 0.0 if math.isnan(ty_std) else ty_std
        ty_ver_m = 0.0 if math.isnan(ty_ver_mean) else ty_ver_mean
        ty_ver_s = 0.0 if math.isnan(ty_ver_std) else ty_ver_std

        # グループ別スコア（利用可能な項目の平均）
        basic_terms: List[float] = []
        if vb_list:
            basic_terms.append(vb_m)
        if tb_list:
            basic_terms.append(tb_m)
        basic_grp = (sum(basic_terms) / len(basic_terms)) if basic_terms else 0.0

        combo_terms: List[float] = []
        if vc_list:
            combo_terms.append(vc_m)
        if tc_list:
            combo_terms.append(tc_m)
        combo_grp = (sum(combo_terms) / len(combo_terms)) if combo_terms else 0.0

        # Typhoon = TrainTyphoonRecallMean + VerTyphoonRecallMean
        typhoon_grp_mean = ty_m + ty_ver_m
        typhoon_grp_std  = ty_s + ty_ver_s

        # Std グループ（存在するものの平均、Typhoon は和）
        basic_std_terms: List[float] = []
        if vb_list:
            basic_std_terms.append(vb_s)
        if tb_list:
            basic_std_terms.append(tb_s)
        basic_grp_std = (sum(basic_std_terms) / len(basic_std_terms)) if basic_std_terms else 0.0

        combo_std_terms: List[float] = []
        if vc_list:
            combo_std_terms.append(vc_s)
        if tc_list:
            combo_std_terms.append(tc_s)
        combo_grp_std = (sum(combo_std_terms) / len(combo_std_terms)) if combo_std_terms else 0.0

        # 総合スコア（Mean/Std）をそれぞれ単純加算
        score_mean = basic_grp + combo_grp + typhoon_grp_mean
        score_std  = basic_grp_std + combo_grp_std + typhoon_grp_std

        # BestIter/BestSeed の決定（Overall のときは (iter, seed) を横断で評価）
        best_iter: str = context_name
        best_seed: Optional[int] = None
        if context_name == "Overall" and roots_data:
            maps_rs: List[Dict[Tuple[str, int], float]] = []
            for root_path, agg_root in roots_data.items():
                root_label = os.path.basename(root_path)
                vb_pairs = agg_root.get(method, {}).get("ver_basic_pairs", [])  # type: ignore[arg-type]
                vc_pairs = agg_root.get(method, {}).get("ver_combo_pairs", [])  # type: ignore[arg-type]
                tb_pairs = agg_root.get(method, {}).get("basic_pairs", [])      # type: ignore[arg-type]
                tc_pairs = agg_root.get(method, {}).get("combo_pairs", [])      # type: ignore[arg-type]
                nw_pairs = agg_root.get(method, {}).get("nodewise_pairs", [])   # type: ignore[arg-type]
                if vb_pairs: maps_rs.append(pair_means_by_root_seed(vb_pairs, root_label))
                if tb_pairs: maps_rs.append(pair_means_by_root_seed(tb_pairs, root_label))
                if vc_pairs: maps_rs.append(pair_means_by_root_seed(vc_pairs, root_label))
                if tc_pairs: maps_rs.append(pair_means_by_root_seed(tc_pairs, root_label))
                if nw_pairs: maps_rs.append(pair_means_by_root_seed(nw_pairs, root_label))
            # Typhoon per (root,seed)
            ty_pairs = agg_root.get(method, {}).get("typhoon_pairs", [])  # type: ignore[arg-type]
            if ty_pairs:
                maps_rs.append(pair_means_by_root_seed(ty_pairs, root_label))
            brs = select_best_root_seed_balanced(maps_rs)
            if brs is not None:
                best_iter, best_seed = brs[0], brs[1]
        if best_seed is None:
            # フォールバック: 集計単位内（iter別またはOverall統合）で seed のみでバランス評価（Basic/Combo/Nodewise/Typhoon）
            vb_map = seed_means_from_pairs(metrics.get("ver_basic_pairs", []))  # type: ignore[arg-type]
            tb_map = seed_means_from_pairs(metrics.get("basic_pairs", []))      # type: ignore[arg-type]
            vc_map = seed_means_from_pairs(metrics.get("ver_combo_pairs", []))  # type: ignore[arg-type]
            tc_map = seed_means_from_pairs(metrics.get("combo_pairs", []))      # type: ignore[arg-type]
            nw_map = seed_means_from_pairs(metrics.get("nodewise_pairs", []))   # type: ignore[arg-type]
            ty_map = seed_means_from_pairs(metrics.get("typhoon_pairs", []))    # type: ignore[arg-type]
            best_seed = select_best_seed_balanced([vb_map, tb_map, vc_map, tc_map, nw_map, ty_map])
            if best_seed is None:
                best_iter = context_name
        rows.append((method, score_mean, score_std, vb_mean, vb_std, vc_mean, vc_std, tb_mean, tb_std, tc_mean, tc_std, nw_mean, nw_std, ty_mean, ty_std, ty_ver_mean, ty_ver_std, best_iter, best_seed, len(vb_list)))

    # スコア降順で上位
    rows.sort(key=lambda x: (-x[1], x[0]))

    print(f"==== 総合推奨手法（暫定スコアに基づく）[{context_name}] ====")
    header = (
        f'{"Method":24s} {"Score(SumMean)":>16s} {"Score(SumStd)":>15s} '
        f'{"VerBasicMean":>13s} {"VerBasicStd":>12s} {"VerComboMean":>13s} {"VerComboStd":>12s} '
        f'{"TrainBasicMean":>15s} {"TrainBasicStd":>14s} {"TrainComboMean":>15s} {"TrainComboStd":>14s} '
        f'{"NodewiseOverallMean":>20s} {"NodewiseOverallStd":>19s} '
        f'{"TyphoonTrainMean":>17s} {"TyphoonTrainStd":>16s} {"TyphoonVerMean":>15s} {"TyphoonVerStd":>14s} '
        f'{"BestIter":>12s} {"BestSeed":>8s} {"N(VerB)":>8s}'
    )
    print(header)
    print("-" * len(header))
    k = len(rows) if (topk is None or topk <= 0 or topk > len(rows)) else topk
    for i, (method, score_mean, score_std, vb_mean, vb_std, vc_mean, vc_std, tb_mean, tb_std, tc_mean, tc_std, nw_mean, nw_std, ty_mean, ty_std, ty_ver_mean, ty_ver_std, best_iter, best_seed, n_vb) in enumerate(rows[:k]):
        print(
            f"{method:24s} {fmt_float(score_mean, prec):>16s} {fmt_float(score_std, prec):>15s} "
            f"{fmt_float(vb_mean, prec):>13s} {fmt_float(vb_std, prec):>12s} {fmt_float(vc_mean, prec):>13s} {fmt_float(vc_std, prec):>12s} "
            f"{fmt_float(tb_mean, prec):>15s} {fmt_float(tb_std, prec):>14s} {fmt_float(tc_mean, prec):>15s} {fmt_float(tc_std, prec):>14s} "
            f"{fmt_float(nw_mean, prec):>20s} {fmt_float(nw_std, prec):>19s} "
            f"{fmt_float(ty_mean, prec):>17s} {fmt_float(ty_std, prec):>16s} {fmt_float(ty_ver_mean, prec):>15s} {fmt_float(ty_ver_std, prec):>14s} "
            f"{best_iter:>12s} {fmt_seed(best_seed):>8s} {n_vb:8d}"
        )
    print("")
    print("注: スコアは Mean と Std を別集計した2列を表示します（いずれも単純和）。")
    print("  Score(SumMean) = BasicMean + ComboMean + (TrainTyphoonRecallMean + VerTyphoonRecallMean)")
    print("    - BasicMean = mean({VerBasicMean, TrainBasicMean} 利用可能のみ)")
    print("    - ComboMean = mean({VerComboMean, TrainComboMean} 利用可能のみ)")
    print("  Score(SumStd)  = BasicStd  + ComboStd  + (TrainTyphoonRecallStd + VerTyphoonRecallStd)")
    print("    - BasicStd  = mean({VerBasicStd,  TrainBasicStd} 利用可能のみ)")
    print("    - ComboStd  = mean({VerComboStd,  TrainComboStd} 利用可能のみ)")
    print("  Nodewise系は参考指標でスコアには不使用です。")
    print("")


def maybe_write_csv(csv_path: Optional[str], per_root_aggregates: Dict[str, Dict[str, Dict[str, Any]]], overall_agg: Dict[str, Dict[str, Any]], prec: int):
    """
    横断ピボット（Verification Basic の平均）の CSV を出力
    カラム: method, <root1>, <root2>, ..., overall_mean, overall_std, n_all
    """
    if not csv_path:
        return
    roots = list(per_root_aggregates.keys())
    roots.sort()
    methods = set()
    for agg in per_root_aggregates.values():
        methods |= set(agg.keys())
    methods = sorted(methods, key=lambda m: -mean_or_nan(overall_agg.get(m, {}).get("ver_basic", [])))  # type: ignore[arg-type]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["method"] + [os.path.basename(r) for r in roots] + ["overall_mean", "overall_std", "n_all"]
        writer.writerow(header)
        for method in methods:
            row = [method]
            vb_all = overall_agg.get(method, {}).get("ver_basic", [])  # type: ignore[arg-type]
            overall_mean_v = mean_or_nan(vb_all)
            overall_std_v = std_or_nan(vb_all)
            n_all = len(vb_all)
            for r in roots:
                agg = per_root_aggregates[r]
                vals = agg.get(method, {}).get("ver_basic", [])  # type: ignore[arg-type]
                row.append(fmt_float(mean_or_nan(vals), prec))
            row += [fmt_float(overall_mean_v, prec), fmt_float(overall_std_v, prec), n_all]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="複数 results_* ディレクトリのログから手法別の各種統計（学習/検証）を算出（横断・総合評価対応）")
    # 既定では src/PressurePattern/ 配下の results_v6_iter100/1000/10000 を自動探索（存在するものだけ）
    default_dir = os.path.dirname(os.path.abspath(__file__))
    default_candidates = [
        os.path.join(default_dir, "results_v6_iter100"),
        os.path.join(default_dir, "results_v6_iter1000"),
        os.path.join(default_dir, "results_v6_iter10000"),
    ]
    default_roots = [p for p in default_candidates if os.path.isdir(p)]
    default_root_single = default_roots[0] if default_roots else os.path.join(default_dir, "results_v6_iter100")

    parser.add_argument(
        "--roots",
        type=str,
        nargs="+",
        default=default_roots,
        help=f"探索対象の results ディレクトリを複数指定（空なら既定候補を自動使用）"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=f"単一の results ディレクトリ（後方互換用途、--roots が指定されていれば無視）"
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="rank",
        choices=["rank", "name", "basic_combo"],
        help="表示順のソートキー: rank(各表で平均降順) / name(名前昇順) / basic_combo(基本/基本+応用を各平均降順)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="小数点以下の表示桁数 (default: 3)",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="総合スコアに基づく推奨手法 Top-K を表示"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="--recommend 使用時の推奨件数。0 以下で全件表示 (default: 0=all)"
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="横断ピボット（Verification Basic の平均）の CSV 出力先"
    )
    parser.add_argument(
        "--hide-overall",
        action="store_true",
        help="Overall（全ルート統合）表を非表示にする（ルート別のみ表示）"
    )
    args = parser.parse_args()

    # roots 決定（--roots 優先、未指定なら --root を使用、いずれも無ければ既定候補/単一）
    roots: List[str] = []
    if args.roots:
        roots = args.roots
    elif args.root:
        roots = [args.root]
    else:
        roots = default_roots if default_roots else [default_root_single]

    # 実在チェック
    roots = [r for r in roots if os.path.isdir(r)]
    if not roots:
        print(f"[ERROR] 指定のディレクトリが存在しません。--roots / --root を確認してください。")
        return

    # 収集
    per_root_eval_counts: Dict[str, int] = {}
    per_root_ver_counts: Dict[str, int] = {}
    per_root_aggregates: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in roots:
        eval_paths, ver_paths, agg = collect_logs(r)
        per_root_eval_counts[r] = len(eval_paths)
        per_root_ver_counts[r] = len(ver_paths)
        per_root_aggregates[r] = agg

    # 統合
    overall_agg = merge_aggregates(list(per_root_aggregates.values()))

    print("==== results 集計（横断） ====")
    print("探索対象（roots）:")
    for r in roots:
        print(f" - {r} (evaluation={per_root_eval_counts[r]} verification={per_root_ver_counts[r]})")
    print("")

    prec = args.precision
    sort_mode = args.sort

    # ルート別の表（例: results_v6_iter100 / 1000 / 10000）
    for r in roots:
        ctx = os.path.basename(r)
        print_all_tables_for_aggregate(per_root_aggregates[r], ctx, sort_mode, prec)
        if args.recommend:
            recommend_methods(per_root_aggregates[r], args.topk, prec, context_name=ctx)

    # Overall（全 roots 統合）の表（必要に応じて表示）
    if not args.hide_overall:
        print_overall_tables(overall_agg, sort_mode, prec)

    # 追加: 横断ピボット表（Verification Basic の平均）
    print_cross_root_ver_basic(per_root_aggregates, overall_agg, prec)

    # 推奨手法
    if args.recommend:
        recommend_methods(overall_agg, args.topk, prec, context_name="Overall", roots_data=per_root_aggregates)

    # CSV 出力
    maybe_write_csv(args.csv_out, per_root_aggregates, overall_agg, prec)

    print("注記:")
    print(" - 学習(評価ログ)/検証(verification)の Macro Recall は [Summary] の値から算出（Mean/Min/Median/Max）。")
    print(" - 各表では MinSeed / MaxSeed に、最小/最大値が出た seed を表示（同値が複数ある場合は最初に検出したもの）。")
    print(" - 6A/6B/6C の Correct/Recall は「各ラベルの再現率（代表ノード群ベース）」の値を使用（学習評価ログ）。")
    print(" - 横断ピボット表では各 Root（ディレクトリ）ごとの検証 Basic 平均と、Overall の平均/標準偏差/件数を確認できます。")
    print(" - --recommend で総合スコア（Basic系: VerBasic/TrainBasic の Mean 平均 + Combo系: VerCombo/TrainCombo の Mean 平均 + Typhoon系: TrainTyphoonRecallMean + VerTyphoonRecallMean）による上位手法を出力します。")
    print(" - 並び順は --sort オプションで制御可能。--precision で表示桁数を調整できます。")


if __name__ == "__main__":
    main()

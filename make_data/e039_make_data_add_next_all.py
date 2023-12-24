NOTEBOOK_NAME = "e039_make_data_add_next_all"

import os


class CFG:
    seed = 127
    fold_num = 5
    OUTPUT_DIR = f"saved_data/{NOTEBOOK_NAME}"
    SEED = 33
    TARGET_COL = "reserve"


os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import GroupKFold, KFold

train_log = pl.read_csv("data/train_log.csv")
label = pl.read_csv("data/train_label.csv")
test_log = pl.read_csv("data/test_log.csv")
yado = pl.read_csv("data/yado.csv")

import pickle

# e016にて、データ作成時にfoldを利用するように変更
with open(
    "saved_data/e016_make_train_popular_base/session_id_fold_dict.pkl", "rb"
) as f:
    session_id_fold_dict = pickle.load(f)

label = label.with_columns(
    label.get_column("session_id").map_dict(session_id_fold_dict).alias("fold")
)


def create_past_view_yado_candidates(log):
    """
    アクセスした宿をcandidateとして作成。ただし、直近の宿は予約しないので除外する。
    """
    max_seq_no = log.group_by("session_id").agg(pl.max("seq_no").alias("max_seq_no"))
    log = log.join(max_seq_no, on="session_id")
    # 最大値に該当する行を除外する
    past_yado_candidates = log.filter(pl.col("seq_no") != pl.col("max_seq_no"))
    past_yado_candidates = past_yado_candidates.select(
        ["session_id", "yad_no"]
    ).unique()

    # 簡易的な特徴量も作成しておく。
    # 何個前に見たか 複数回見た時は、直近のみ残す。
    past_yado_feature = log.with_columns(
        (pl.col("max_seq_no") - pl.col("seq_no")).alias("max_seq_no_diff")
    ).filter(pl.col("seq_no") != pl.col("max_seq_no"))
    past_yado_feature = past_yado_feature.join(
        past_yado_feature.group_by(["session_id", "yad_no"]).agg(
            pl.col("max_seq_no_diff").max().alias("max_seq_no_diff")
        ),
        on=["session_id", "yad_no", "max_seq_no_diff"],
    )
    # 何回見たか
    session_view_count = (
        log.group_by(["session_id", "yad_no"])
        .count()
        .rename({"count": "session_view_count"})
    )
    past_yado_feature = past_yado_feature.join(
        session_view_count, how="left", on=["session_id", "yad_no"]
    ).drop("seq_no")

    return past_yado_candidates, past_yado_feature


def create_topN_popular_yado_candidates(label, train_test="train", top=10):
    """
    予約された人気宿をcandidateとして作成。train/validでリークしないように注意。
    """
    # labelデータを使うので、学習データはtrain/validで分割して作成。
    top10_yado_candidate = pl.DataFrame()
    popular_yado_feature = pl.DataFrame()
    if train_test == "train":
        for fold in range(CFG.fold_num):
            train_label = label.filter(pl.col("fold") != fold)
            popular_yado_sort = (
                train_label["yad_no"].value_counts().sort(by="counts", descending=True)
            )

            # candidateの作成
            top10_yado_candidate_fold = (
                popular_yado_sort.head(top)
                .with_columns(pl.lit(fold).alias("fold"))
                .select(["yad_no", "fold"])
            )
            top10_yado_candidate = pl.concat(
                [top10_yado_candidate, top10_yado_candidate_fold]
            )

            # 簡易的な特徴量も作成しておく。
            popular_yado_feature_fold = popular_yado_sort.with_columns(
                pl.lit(fold).alias("fold")
            )
            popular_yado_feature_fold = popular_yado_feature_fold.with_columns(
                pl.arange(1, len(popular_yado_sort) + 1).alias("popular_rank")
            )
            popular_yado_feature = pl.concat(
                [popular_yado_feature, popular_yado_feature_fold]
            )
    else:  # testデータはtrainデータ全体で作成する。
        # candidateの作成
        popular_yado_sort = (
            label["yad_no"].value_counts().sort(by="counts", descending=True)
        )
        top10_yado_candidate = popular_yado_sort.head(top).select(["yad_no"])

        # 簡易的な特徴量も作成しておく。
        popular_yado_feature = popular_yado_sort.with_columns(
            pl.arange(1, len(popular_yado_sort) + 1).alias("popular_rank")
        )

    popular_yado_feature = popular_yado_feature.rename({"counts": "reservation_counts"})

    return top10_yado_candidate, popular_yado_feature


def create_topN_area_popular_yado_candidates(
    label, yado, train_test="train", area="wid_cd", top=10
):
    """
    エリア単位で予約された人気宿をcandidateとして作成。train/validでリークしないように注意。
    """
    label_yado = label.join(yado, how="left", on="yad_no")
    # labelデータを使うので、学習データはtrain/validで分割して作成。
    top10_yado_area_candidate = pl.DataFrame()
    popular_yado_area_feature = pl.DataFrame()
    if train_test == "train":
        for fold in range(CFG.fold_num):
            train_label = label.filter(pl.col("fold") != fold)
            popular_yado_sort = (
                label_yado.group_by([area, "yad_no"])
                .count()
                .sort(by=[area, "count"], descending=[False, True])
            )

            # candidateの作成
            top10_yado_area_candidate_fold = (
                popular_yado_sort.group_by(area)
                .head(top)
                .with_columns(pl.lit(fold).alias("fold"))
                .select([area, "yad_no", "fold"])
            )
            top10_yado_area_candidate = pl.concat(
                [top10_yado_area_candidate, top10_yado_area_candidate_fold]
            )

            # 簡易的な特徴量も作成しておく。
            popular_yado_area_feature_fold = popular_yado_sort.with_columns(
                pl.lit(fold).alias("fold")
            )
            popular_yado_area_feature_fold = popular_yado_area_feature_fold.group_by(
                area
            ).map_groups(
                lambda group: group.with_columns(
                    pl.col("count")
                    .rank(method="dense", descending=True)
                    .over(area)
                    .alias(f"popular_{area}_rank")
                )
            )
            popular_yado_area_feature = pl.concat(
                [popular_yado_area_feature, popular_yado_area_feature_fold]
            )

    else:  # testデータはtrainデータ全体で作成する。
        # candidateの作成
        popular_yado_sort = (
            label_yado.group_by([area, "yad_no"])
            .count()
            .sort(by=[area, "count"], descending=[False, True])
        )
        top10_yado_area_candidate = (
            popular_yado_sort.group_by(area).head(top).select([area, "yad_no"])
        )

        # 簡易的な特徴量も作成しておく。
        popular_yado_area_feature = popular_yado_sort.group_by(area).map_groups(
            lambda group: group.with_columns(
                pl.col("count")
                .rank(method="dense", descending=True)
                .over(area)
                .alias(f"popular_{area}_rank")
            )
        )

    popular_yado_area_feature = popular_yado_area_feature.drop("count")

    return top10_yado_area_candidate, popular_yado_area_feature


def create_latest_next_booking_tonN_candidate(log, label, train_test="train", top=10):
    """
    直近見た宿で、次にどこを予約しやすいか。
    """
    log_latest = train_log.group_by("session_id").tail(1)
    log_latest = log_latest.rename({"yad_no": "latest_yad_no"})
    log_latest = log_latest.join(label, how="left", on="session_id")

    # labelデータを使うので、学習データはtrain/validで分割して作成。
    latest_next_booking_tonN_candidate = pl.DataFrame()
    latest_next_booking_tonN_feature = pl.DataFrame()
    if train_test == "train":
        for fold in range(CFG.fold_num):
            train_log_latest = log_latest.filter(pl.col("fold") != fold)
            train_log_latest = (
                train_log_latest.group_by(["latest_yad_no", "yad_no"])
                .count()
                .sort(by=["latest_yad_no", "count"], descending=[False, True])
            )

            # candidateの作成
            latest_next_booking_tonN_candidate_fold = (
                train_log_latest.group_by("latest_yad_no")
                .head(top)
                .with_columns(pl.lit(fold).alias("fold"))
                .select(["yad_no", "latest_yad_no", "fold"])
            )
            latest_next_booking_tonN_candidate = pl.concat(
                [
                    latest_next_booking_tonN_candidate,
                    latest_next_booking_tonN_candidate_fold,
                ]
            )

            # 簡易的な特徴量も作成しておく。
            latest_next_booking_tonN_feature_fold = train_log_latest.with_columns(
                pl.lit(fold).alias("fold")
            )
            latest_next_booking_tonN_feature_fold = (
                latest_next_booking_tonN_feature_fold.group_by(
                    "latest_yad_no"
                ).map_groups(
                    lambda group: group.with_columns(
                        pl.col("count")
                        .rank(method="dense", descending=True)
                        .over("latest_yad_no")
                        .alias(f"latest_next_booking_rank")
                    )
                )
            )
            latest_next_booking_tonN_feature = pl.concat(
                [
                    latest_next_booking_tonN_feature,
                    latest_next_booking_tonN_feature_fold,
                ]
            )
    else:
        log_latest = (
            log_latest.group_by(["latest_yad_no", "yad_no"])
            .count()
            .sort(by=["latest_yad_no", "count"], descending=[False, True])
        )

        # candidateの作成
        latest_next_booking_tonN_candidate = (
            log_latest.group_by("latest_yad_no")
            .head(top)
            .select(["yad_no", "latest_yad_no"])
        )

        # 簡易的な特徴量も作成しておく。
        latest_next_booking_tonN_feature = log_latest.group_by(
            "latest_yad_no"
        ).map_groups(
            lambda group: group.with_columns(
                pl.col("count")
                .rank(method="dense", descending=True)
                .over("latest_yad_no")
                .alias(f"latest_next_booking_rank")
            )
        )
    latest_next_booking_tonN_feature = latest_next_booking_tonN_feature.drop("count")
    return latest_next_booking_tonN_candidate, latest_next_booking_tonN_feature


(
    train_past_view_yado_candidates,
    train_past_view_yado_feature,
) = create_past_view_yado_candidates(train_log)
(
    test_past_view_yado_candidates,
    test_past_view_yado_feature,
) = create_past_view_yado_candidates(test_log)

(
    train_top10_popular_yado_candidates,
    train_top10_popular_yado_feature,
) = create_topN_popular_yado_candidates(label, train_test="train", top=10)
(
    test_top10_popular_yado_candidates,
    test_top10_popular_yado_feature,
) = create_topN_popular_yado_candidates(label, train_test="test", top=10)

# (
#     train_top10_wid_popular_yado_candidates,
#     train_top10_wid_popular_yado_feature,
# ) = create_topN_area_popular_yado_candidates(
#     label, yado, train_test="train", area="wid_cd", top=10
# )
# (
#     test_top10_wid_popular_yado_candidates,
#     test_top10_wid_popular_yado_feature,
# ) = create_topN_area_popular_yado_candidates(
#     label, yado, train_test="test", area="wid_cd", top=10
# )

# (
#     train_top10_ken_popular_yado_candidates,
#     train_top10_ken_popular_yado_feature,
# ) = create_topN_area_popular_yado_candidates(
#     label, yado, train_test="train", area="ken_cd", top=10
# )
# (
#     test_top10_ken_popular_yado_candidates,
#     test_top10_ken_popular_yado_feature,
# ) = create_topN_area_popular_yado_candidates(
#     label, yado, train_test="test", area="ken_cd", top=10
# )

(
    train_top10_lrg_popular_yado_candidates,
    train_top10_lrg_popular_yado_feature,
) = create_topN_area_popular_yado_candidates(
    label, yado, train_test="train", area="lrg_cd", top=10
)
(
    test_top10_lrg_popular_yado_candidates,
    test_top10_lrg_popular_yado_feature,
) = create_topN_area_popular_yado_candidates(
    label, yado, train_test="test", area="lrg_cd", top=10
)

(
    train_top10_sml_popular_yado_candidates,
    train_top10_sml_popular_yado_feature,
) = create_topN_area_popular_yado_candidates(
    label, yado, train_test="train", area="sml_cd", top=10
)
(
    test_top10_sml_popular_yado_candidates,
    test_top10_sml_popular_yado_feature,
) = create_topN_area_popular_yado_candidates(
    label, yado, train_test="test", area="sml_cd", top=10
)

(
    train_latest_next_booking_ton10_candidate,
    train_latest_next_booking_ton10_feature,
) = create_latest_next_booking_tonN_candidate(
    train_log, label, train_test="train", top=10
)
(
    test_latest_next_booking_ton10_candidate,
    test_latest_next_booking_ton10_feature,
) = create_latest_next_booking_tonN_candidate(
    train_log, label, train_test="test", top=100_000
)

train_past_view_yado_candidates.write_parquet(
    f"{CFG.OUTPUT_DIR}/train_past_view_yado_candidates.parquet"
)
train_past_view_yado_feature.write_parquet(
    f"{CFG.OUTPUT_DIR}/train_past_view_yado_feature.parquet"
)
test_past_view_yado_candidates.write_parquet(
    f"{CFG.OUTPUT_DIR}/test_past_view_yado_candidates.parquet"
)
test_past_view_yado_feature.write_parquet(
    f"{CFG.OUTPUT_DIR}/test_past_view_yado_feature.parquet"
)

# train_top10_popular_yado_candidates.write_parquet(
#     f"{CFG.OUTPUT_DIR}/train_top10_popular_yado_candidates.parquet"
# )
# train_top10_popular_yado_feature.write_parquet(
#     f"{CFG.OUTPUT_DIR}/train_top10_popular_yado_feature.parquet"
# )
# test_top10_popular_yado_candidates.write_parquet(
#     f"{CFG.OUTPUT_DIR}/test_top10_popular_yado_candidates.parquet"
# )
# test_top10_popular_yado_feature.write_parquet(
#     f"{CFG.OUTPUT_DIR}/test_top10_popular_yado_feature.parquet"
# )

# train_top10_wid_popular_yado_candidates.write_parquet(
#     "candidate/train_top10_wid_popular_yado_candidates.parquet"
# )
# train_top10_wid_popular_yado_feature.write_parquet(
#     "features/train_top10_wid_popular_yado_feature.parquet"
# )
# test_top10_wid_popular_yado_candidates.write_parquet(
#     "candidate/test_top10_wid_popular_yado_candidates.parquet"
# )
# test_top10_wid_popular_yado_feature.write_parquet(
#     "features/test_top10_wid_popular_yado_feature.parquet"
# )

# train_top10_ken_popular_yado_candidates.write_parquet(
#     "candidate/train_top10_ken_popular_yado_candidates.parquet"
# )
# train_top10_ken_popular_yado_feature.write_parquet(
#     "features/train_top10_ken_popular_yado_feature.parquet"
# )
# test_top10_ken_popular_yado_candidates.write_parquet(
#     "candidate/test_top10_ken_popular_yado_candidates.parquet"
# )
# test_top10_ken_popular_yado_feature.write_parquet(
#     "features/test_top10_ken_popular_yado_feature.parquet"
# )

train_top10_lrg_popular_yado_candidates.write_parquet(
    f"{CFG.OUTPUT_DIR}/train_top10_lrg_popular_yado_candidates.parquet"
)
train_top10_lrg_popular_yado_feature.write_parquet(
    f"{CFG.OUTPUT_DIR}/train_top10_lrg_popular_yado_feature.parquet"
)
test_top10_lrg_popular_yado_candidates.write_parquet(
    f"{CFG.OUTPUT_DIR}/test_top10_lrg_popular_yado_candidates.parquet"
)
test_top10_lrg_popular_yado_feature.write_parquet(
    f"{CFG.OUTPUT_DIR}/test_top10_lrg_popular_yado_feature.parquet"
)


train_top10_sml_popular_yado_candidates.write_parquet(
    f"{CFG.OUTPUT_DIR}/train_top10_sml_popular_yado_candidates.parquet"
)
train_top10_sml_popular_yado_feature.write_parquet(
    f"{CFG.OUTPUT_DIR}/train_top10_sml_popular_yado_feature.parquet"
)
test_top10_sml_popular_yado_candidates.write_parquet(
    f"{CFG.OUTPUT_DIR}/test_top10_sml_popular_yado_candidates.parquet"
)
test_top10_sml_popular_yado_feature.write_parquet(
    f"{CFG.OUTPUT_DIR}/test_top10_sml_popular_yado_feature.parquet"
)

train_latest_next_booking_ton10_candidate.write_parquet(
    f"{CFG.OUTPUT_DIR}/train_latest_next_booking_top10_candidates.parquet"
)
train_latest_next_booking_ton10_feature.write_parquet(
    f"{CFG.OUTPUT_DIR}/train_latest_next_booking_top10_feature.parquet"
)
test_latest_next_booking_ton10_candidate.write_parquet(
    f"{CFG.OUTPUT_DIR}/test_latest_next_booking_top10_candidates.parquet"
)
test_latest_next_booking_ton10_feature.write_parquet(
    f"{CFG.OUTPUT_DIR}/test_latest_next_booking_top10_feature.parquet"
)

# candidateの結合

# area単位のは多すぎるので、今回は除外。
candidate_name_list = [
    "past_view_yado",
    # 'top20_popular_yado',
    # 'top10_wid_popular_yado',
    # 'top10_ken_popular_yado',
    # "top10_lrg_popular_yado",
    # "top10_sml_popular_yado",
    "latest_next_booking_top10",
]


def get_session_id_list(log):
    return log.group_by("session_id").head(1).select(["session_id"])


train_session_id = get_session_id_list(train_log)
train_session_id = train_session_id.join(
    label.select(["fold", "session_id"]), how="left", on="session_id"
)

test_session_id = get_session_id_list(test_log)

from tqdm import tqdm

# 各candidateを結合
candidate_list = {}
candidate_list["train"] = []
candidate_list["test"] = []

for train_test in ["train", "test"]:
    for candidate_name in tqdm(candidate_name_list):
        candidate = pl.read_parquet(
            f"{CFG.OUTPUT_DIR}/{train_test}_{candidate_name}_candidates.parquet"
        )
        if "session_id" in candidate.columns:
            candidate_list[train_test].append(
                candidate.select(["session_id", "yad_no"])
            )
        elif "latest_yad_no" in candidate.columns:
            if train_test == "train":
                latest_yad_no = (
                    train_log.group_by("session_id")
                    .tail(1)
                    .select(["session_id", "yad_no"])
                    .rename({"yad_no": "latest_yad_no"})
                )
                latest_yad_no = latest_yad_no.join(
                    label.select(["session_id", "fold"]), how="left", on="session_id"
                )
                latest_yad_no = latest_yad_no.with_columns(
                    pl.col("fold").cast(pl.Int32)
                )
                candidate = latest_yad_no.join(
                    candidate, how="inner", on=["latest_yad_no", "fold"]
                )
            else:
                latest_yad_no = (
                    test_log.group_by("session_id")
                    .tail(1)
                    .select(["session_id", "yad_no"])
                    .rename({"yad_no": "latest_yad_no"})
                )
                candidate = latest_yad_no.join(
                    candidate, how="inner", on=["latest_yad_no"]
                )
            candidate_list[train_test].append(
                candidate.select(["session_id", "yad_no"])
            )

        else:
            if train_test == "train":
                if "fold" in candidate.columns:
                    candidate_all = pl.DataFrame()
                    for fold in range(CFG.fold_num):
                        candidate_fold = train_session_id.filter(
                            pl.col("fold") == fold
                        ).join(
                            candidate.filter(pl.col("fold") == fold).select(["yad_no"]),
                            how="cross",
                        )
                        candidate_all = pl.concat([candidate_all, candidate_fold])
            else:
                candidate_all = test_session_id.join(
                    candidate.select(["yad_no"]), how="cross"
                )
            candidate_list[train_test].append(
                candidate_all.select(["session_id", "yad_no"])
            )

train_candidate = pl.concat(candidate_list["train"]).unique()
test_candidate = pl.concat(candidate_list["test"]).unique()

train_candidate.write_parquet(f"{CFG.OUTPUT_DIR}/train_candidate.parquet")

test_candidate.write_parquet(f"{CFG.OUTPUT_DIR}/test_candidate.parquet")

pass

# # pass
# train_all = []
# test_all = []
# for train_test in ["train", "test"]:
#     for candidate_name in tqdm(candidate_name_list):
#         candidate = pl.read_parquet(
#             f"{CFG.OUTPUT_DIR}/{train_test}_{candidate_name}_candidates.parquet"
#         )
#         if train_test == "train":
#             train_all.append(candidate)
#         else:
#             test_all.append(candidate)

# pass

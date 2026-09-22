"""宮島の紙の選手コメントを数にする。

辞書と読み方は `docs/design_paper_comment.md` で**スコアを1回も出す前に**固定した。
ここはそれをそのまま実装しただけや。辞書をいじるときは設計書も直す。

🚨 否定を先に潰す。「合ってない」は「合って」を含むし、
   「悪くない」は弱い語やのうて強い語や。素朴な部分一致は符号を逆にする。
   → [[insight_same_word_body_or_part]]
"""
from __future__ import annotations

import re
import sqlite3

# --- 否定の潰し ------------------------------------------------------------
# 先にここで「語+否定」を専用の印に置き換える。置換の順番に意味がある。
NEG_RULES: list[tuple[re.Pattern[str], str]] = [
    # 弱い語の否定 = 強い
    (re.compile(r"悪く(ない|なかった|ありません|無い)"), "\x01"),   # 悪くない → +
    (re.compile(r"(重|弱|硬)く(ない|なかった|ありません|無い)"), "\x01"),
    # 強い語の否定 = 弱い
    (re.compile(r"合って(ない|いない|ません|へん|無い)"), "\x02"),
    (re.compile(r"合わ(ない|ん|ず)"), "\x02"),
    (re.compile(r"良く(ない|ありません|無い)"), "\x02"),
    (re.compile(r"上がって(ない|いない|ません|来ない|こない)"), "\x02"),
    (re.compile(r"(出|伸び)て(ない|いない|ません)"), "\x02"),
]

# --- 辞書 v2(2026-09-23 に差し替え。**着順は1行も見とらん**) ----------------
# v1 は強弱ゼロが 35.7% 出た。中身を読んだら「中堅レベル」「上向いた」「もう一足」
# 「重たい」みたいな常用語がまるごと抜けとった=**語彙の穴**で、着順とは無関係に分かる。
# 入力を読んでの穴埋めは事後選択やない。判定は **v2 一本**でやる(両方で判定したら2回検定)。
POS = ["良い", "いい", "上位", "一番", "力強", "仕上が", "バランス",
       "合って", "上がって", "上向い", "十分", "強め", "効果はあった", "\x01"]
NEG = ["今一つ", "イマイチ", "今ひとつ", "悪い", "弱い", "物足り",
       "乗り難", "掛からな", "上がらな", "重い", "重た", "足りな",
       "もう一足", "失敗", "\x02"]
# 「中堅」は宮島の紙で一番よう出る言い回しで、意味は「並」。強でも弱でもない
NEUTRAL = ["普通", "可もなく", "中堅", "変わらず", "変わっていな", "他艇と同じ"]
# 🚨 入れんかった語と理由:
#   「まずまず」「もう少し」= 向きが文脈で割れる(「もう少し調整する」は弱さやない)
#   「調整」「点検」「整備」= 行為であって評価やない。全艇に出る=旗が全部に立つのと同じ
#     → [[insight_missing_reads_as_zero]]


def normalize(text: str) -> str:
    t = text or ""
    for rx, mark in NEG_RULES:
        t = rx.sub(mark, t)
    return t


def score(text: str) -> dict[str, float | int | None]:
    """1艇ぶんのコメントを数にする。空文字は None を返す(0 で埋めん)。"""
    if not text or not text.strip():
        return {"cmt_score": None, "cmt_pos": None, "cmt_neg": None,
                "cmt_neu": None, "cmt_len": None}
    t = normalize(text)
    p = sum(t.count(w) for w in POS)
    n = sum(t.count(w) for w in NEG)
    u = sum(t.count(w) for w in NEUTRAL)
    return {"cmt_score": p - n, "cmt_pos": p, "cmt_neg": n,
            "cmt_neu": u, "cmt_len": len(text)}


def load(con: sqlite3.Connection, jcd: str = "17"):
    """paper_comment を読んで、レース内順位まで付けて返す(pandas)。"""
    import pandas as pd

    df = pd.read_sql_query(
        "SELECT date, jcd, rno, lane, reg, comment FROM paper_comment WHERE jcd=?",
        con, params=(jcd,))
    if df.empty:
        return df
    sc = df["comment"].map(score).apply(pd.Series)
    df = pd.concat([df, sc], axis=1)
    key = ["date", "jcd", "rno"]
    # レース内順位。**欠測は順位を付けん**(NaN のまま残す)
    for col in ("cmt_score", "cmt_len"):
        df[col + "_rank"] = df.groupby(key)[col].rank(method="average", na_option="keep")
    # そのレースで何艇ぶん取れとるか。1〜5艇しか無いレースは順位の意味が変わる
    df["cmt_n_in_race"] = df.groupby(key)["cmt_score"].transform(lambda s: s.notna().sum())
    return df


if __name__ == "__main__":
    import sys

    con = sqlite3.connect(sys.argv[1] if len(sys.argv) > 1 else "data/kyotei.db")
    df = load(con)
    if df.empty:
        print("paper_comment が空や")
        raise SystemExit(0)
    n = len(df)
    print(f"行 {n}  日 {df['date'].nunique()}  レース {df.groupby(['date','rno']).ngroups}")
    print(f"非欠測 cmt_score = {df['cmt_score'].notna().sum()} ({df['cmt_score'].notna().mean():.1%})")
    print(f"\ncmt_score の分布:")
    print(df["cmt_score"].value_counts().sort_index().to_string())
    print(f"\n6艇そろっとるレース: {(df['cmt_n_in_race']==6).sum()//6} / {df.groupby(['date','rno']).ngroups}")
    print(f"\n中立語だけ(強弱ゼロ)の艇: {((df.cmt_pos==0)&(df.cmt_neg==0)).sum()} ({((df.cmt_pos==0)&(df.cmt_neg==0)).mean():.1%})")
    print("\n--- 例: score が高い/低い ---")
    for lab, sub in [("高い", df.nlargest(3, "cmt_score")), ("低い", df.nsmallest(3, "cmt_score"))]:
        for _, r in sub.iterrows():
            print(f"  [{lab} {r.cmt_score:+.0f}] {r.comment[:56]}")

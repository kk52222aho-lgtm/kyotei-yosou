"""戦略トーナメントの評価基盤: リーク無し(walk-forward)予測+全特徴+単勝結果を1枚に固める。

エージェントthaが何通りも戦略(=どの艇をどの条件で単勝で買うか)を作って戦わせる為の土台。
毎回モデルを張り直すと死ぬので、各年walk-forward(train=それ以前)で予測を1度だけ計算し、
特徴と結果を付けてpickle化。以後はこの1枚をフィルタするだけ=高速に何千通りも評価できる。
開発期(dev)=2023-2024、審判(holdout)=2025を分離(過学習と選択バイアスの検出用)。

出力: data/strategy_substrate.pkl  例: python -u -m src.build_substrate
"""
from __future__ import annotations

import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS

FEATS = ["racer_class", "motor_2rate", "motor_3rate", "nat_2rate", "nat_win",
         "loc_2rate", "avg_st", "age", "weight", "tenji_time", "tilt",
         "wind_speed", "wave_height", "water_temp"]


def main():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen FROM entries e JOIN payouts p
        ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]

    out = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            if len(g) != 6:
                continue
            g = g.sort_values("p", ascending=False)
            s = g["p"].sum()
            if s <= 0 or pd.isna(g["tansho_lane"].iloc[0]):
                continue
            win = int(g["tansho_lane"].iloc[0])
            conf = float(g["p"].iloc[0]) / s * 100        # レースの本命確信(混沌度)
            for rank, (_, b) in enumerate(g.iterrows(), 1):
                row = {"yr": y, "jcd": str(jcd), "rno": rno, "lane": int(b["lane"]),
                       "p": float(b["p"]) / s * 100,       # レース内正規化勝率%
                       "rank": rank, "conf": conf, "is_lane1": int(b["lane"] == 1),
                       "won": int(int(b["lane"]) == win),
                       "tansho_yen": float(b["tansho_yen"]),   # レース確定単勝払戻(勝った時のret)
                       "reg": str(b["reg"])}
                for f in FEATS:
                    v = b.get(f)
                    row[f] = float(v) if pd.notna(v) and f != "racer_class" else (
                        str(v) if f == "racer_class" else None)
                out.append(row)
    sub = pd.DataFrame(out)
    path = "data/strategy_substrate.pkl"
    sub.to_pickle(path)
    print(f"基盤 {len(sub):,}行 ({sub['yr'].nunique()}年 x 6艇) → {path}")
    print(f"  年別: {sub.groupby('yr').size().to_dict()}")
    print(f"  列: {list(sub.columns)}")


if __name__ == "__main__":
    main()

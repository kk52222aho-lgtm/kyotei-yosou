"""キャッシュ済みBファイルを再パースして motor_no/boat_no を掘り出しキャッシュ（再DLなし）。

現DBはmotor_2rate/boat_2rateの集計だけで生NO列が無い。harnessで生ID特徴を試すため、
data/raw の b*.lzh を全部再パースして (date,jcd,rno,lane,motor_no,boat_no) を
data/motor_boat.csv に吐く。entriesテーブルのスキーマは触らん（実験用サイドキャッシュ）。

例: python -u -m src.backfill_motor_boat
"""
from __future__ import annotations

import glob
import os

import lhafile
import pandas as pd

from . import official, storage

OUT = os.path.join(storage.DATA_DIR, "motor_boat.csv")


def main():
    rows = []
    files = sorted(glob.glob(os.path.join(storage.DATA_DIR, "raw", "b*.lzh")))
    print(f"Bファイル {len(files):,}枚を再パース…")
    for i, path in enumerate(files):
        date = "20" + os.path.basename(path)[1:7]
        try:
            lf = lhafile.Lhafile(path)
            data = lf.read(lf.namelist()[0])
        except Exception:
            continue
        for (jcd, rno), entries in official.parse_b(data).items():
            for e in entries:
                rows.append((date, jcd, int(rno), e["lane"], e.get("motor_no"), e.get("boat_no")))
        if (i + 1) % 500 == 0:
            print(f"  {i+1:,}/{len(files):,}")
    df = pd.DataFrame(rows, columns=["date", "jcd", "rno", "lane", "motor_no", "boat_no"])
    df.to_csv(OUT, index=False)
    print(f"\n保存 {OUT}: {len(df):,}行")
    print(f"ユニーク motor_no {df.motor_no.nunique()} / boat_no {df.boat_no.nunique()} "
          f"/ (場,motor)複合 {df.groupby(['jcd','motor_no']).ngroups:,}")
    print(f"欠損: motor_no {df.motor_no.isna().mean()*100:.1f}% / boat_no {df.boat_no.isna().mean()*100:.1f}%")
    print("サンプル:")
    print(df.head(4).to_string(index=False))


if __name__ == "__main__":
    main()

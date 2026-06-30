# 競艇予想サイト (kyotei-yosou)

boatrace.jp の公式データを収集し、機械学習で各艇の1着確率を予測。
さらに**回収率バックテストで「勝てる形」を検証**し、毎日の妙味レースを抽出する予想サイト。

> ✅ **ステータス: 前進検証クリア（実行リスクは未検証）**。戦略を凍結し、選択に一度も使っていない
> 3年(2023/2024/2025)で前進検証 → **単勝 各年108-121%（各~800レース）/ 2連単3点 181-184%**。
> 選択バイアス・小サンプルの疑いは晴れた。⚠️ ただし確定オッズ前提で**実弾の妥当性(プール薄・オッズ変動)は未検証**。
> 詳細は [docs/FINDINGS.md](docs/FINDINGS.md)、再現は `python -m src.validate`。

---

## クイックスタート

```bash
pip install -r requirements.txt

# 1) データ収集（公式オープンデータ＝高速・全場）
python -m src.official --start 20260101 --end 20260618

# 2) 学習
python -m src.train

# 3) 本日の妙味レースを抽出
python -m src.scan

# 4) Web アプリ起動
streamlit run streamlit_app.py          # Streamlit 版（推奨）→ http://localhost:8501
uvicorn src.app:app --reload            # FastAPI 版 → http://127.0.0.1:8000 /today
```

---

## モジュール一覧

### データ
| ファイル | 役割 |
|---|---|
| `src/official.py` | **公式オープンデータ(B番組表/K競走成績 lzh)** を一括収集。1日2ファイルで全15-24場、払戻金も。主力 |
| `src/scraper.py` | boatrace.jp スクレイパー（出走表/結果/直前情報/単勝・3連単オッズ/開催場）。ライブ予測用 |
| `src/collect.py` | スクレイピングによる収集 CLI（直前情報込み） |
| `src/storage.py` | SQLite 保存（entries 1艇1行 ＋ payouts 払戻金） |
| `src/venues.py` | 場コード↔場名（愛知近郊: 蒲郡07/常滑08/津09） |

### モデル
| ファイル | 役割 |
|---|---|
| `src/features.py` | 特徴量（**場venue**・艇番・級別・勝率/連率・モーター・展示タイム・風 等） |
| `src/train.py` | 1着確率モデル（HistGradientBoosting + 確率校正）。1号艇ベタ買いと必ず比較 |
| `src/predict.py` | ライブ予測・印付け・期待値・**推奨買い目(recommend)**・Harville 3連単確率 |

### 検証
| ファイル | 役割 |
|---|---|
| `src/backtest.py` | 回収率バックテスト（払戻金ベース、7戦略） |
| `src/backtest_trifecta.py` | 3連単 EV バックテスト（オッズ取得→Harville→戦略別ROI） |
| `src/analyze.py` | セグメント分析（条件別の回収率） |
| `src/strategy_search.py` | **「勝てる形」総当たり探索**（全賭式×条件、out-of-sample） |
| `src/simulate.py` | バンクロール・シミュレーション（資金曲線・最大DD・連敗） |
| `src/validate.py` | **最終検証**（全年・年次ウォークフォワード＋頑健性判定） |

### プロダクト
| ファイル | 役割 |
|---|---|
| `src/scan.py` | **本日の妙味レース・スキャナ**（開催全場→本命≠1号艇を抽出→推奨買い目）。JSONキャッシュ |
| `src/app.py` | FastAPI。`/` トップ・`/today` 本日の妙味レース・`/race` 個別予想 |
| `templates/` | index / today / race の各画面 |

---

## 使い方詳細

### データ収集

```bash
# 公式オープンデータ（推奨・全場・高速・払戻込み）
python -m src.official --start 20260101 --end 20260618

# 払戻金のみキャッシュから再生成（再DL不要）
python -c "from src import storage,official; official.backfill_payouts(storage.connect())"

# スクレイピング（本日のライブ用・全項目＋直前情報）
python -m src.collect --start 20260630 --end 20260630 --venues 07 08 09
```

公式データは2012年以降が構造的に一貫（持ちペラ廃止等）。古いデータと混ぜない方針。

### 学習・検証

```bash
python -m src.train              # 学習＋1号艇ベタ買いとの比較
python -m src.strategy_search    # 勝てる形を全賭式×条件で探索
python -m src.simulate           # 資金曲線・ドローダウン
python -m src.validate           # 全年・年次ウォークフォワードで最終判定
```

### 予想・実戦

```bash
python -m src.scan                       # 本日の妙味レース（JSON保存）
python -m src.scan --date 20260630
uvicorn src.app:app --reload             # /today で本日の買い目
```

### 前向きペーパートレード運用（実弾妥当性の実記録）

```bash
python -m src.papertrade log       # 締切前: 本日の買い目＋現オッズを記録
python -m src.papertrade settle    # 確定後: 実結果で精算
python -m src.papertrade report    # 実トラック回収率
```

`run_daily.bat` を Windows タスクスケジューラ（タスク名 `KyoteiPaperTrade`, 毎朝9:00）で自動実行中。
台帳は `data/papertrade.jsonl`、実行ログは `data/papertrade_run.log`。
停止: `schtasks /delete /tn KyoteiPaperTrade /f`。
※log を締切前に走らせたぶんだけが本物の前向き記録（過去日の遡及記録は track に含めない方針）。

---

## 勝てる形（前進検証クリア・実弾は未検証）

**モデル本命がイン(1号艇)以外のレースだけ、その本命を単勝＋2連単(上位3点)で買う。**

戦略を凍結し、選択に使っていない3年で前進検証（学習=テスト年より前のみ・補完リークなし）:

| クリーン年(フル) | 該当 | 単勝 | 2連単3点 |
|---|---|---|---|
| 2023 | 1,254 | 109% | 183% |
| 2024 | 1,225 | 125% | 195% |
| 2025 | 1,183 | 113% | 189% |

> 選択バイアス・小サンプルの疑いは完全に晴れた（3年フル・計3,662ベット・全て100%超）。
> ⚠️ 残る穴: ①単勝プールの実行リスク（自分の賭けでオッズが動く/確定オッズ前提）②2連単+80%の桁は
> 実オッズ紙トレ必須 ③検証は各年9ヶ月分時点。**実弾投入前に実オッズでの紙トレを。**

- なぜ効く: 公衆のイン過剰投票という非効率を突く
- 限界: 賭け頻度は約2.7%（緩めると100%割れ）。単勝はプール小→少額・分散投票
- ❌ 3連単のEV最大化／高ROIは「万舟頼みの分散」＝罠（含めない）

詳細・否定された仮説・利益ラインは [docs/FINDINGS.md](docs/FINDINGS.md)。

---

## 注意

- スクレイピングは公式サーバ配慮のため待機を入れている（`scraper.SLEEP_SEC`）。
- 本ツールは統計的推定であり的中・利益を保証しない。投資は自己責任で。

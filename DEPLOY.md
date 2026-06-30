# デプロイ手順（固定URL・PC不要：Streamlit Community Cloud）

ローカルは git リポジトリ化＋コミット済み（最新モデル同梱）。あとは GitHub に上げて
Streamlit Cloud で公開するだけ。**閲覧はPC不要・固定URL**になる。

---

## 1. GitHub に上げる（要・あなたのGitHub認証）

### 方法A: gh CLI でログイン（対話）
```bash
cd C:\dev\kyotei-yosou
gh auth login          # ブラウザ/デバイスコードで認証（GitHubアカウント）
gh repo create kyotei-yosou --private --source=. --push
```
→ これで private リポジトリが作られ push まで完了。

### 方法B: トークンを使う（釣りプロジェクトと同じ）
GitHubで Personal Access Token (repo権限) を発行済みなら:
```bash
cd C:\dev\kyotei-yosou
git remote add origin https://<TOKEN>@github.com/<ユーザー名>/kyotei-yosou.git
git branch -M main
git push -u origin main
```

---

## 2. Streamlit Community Cloud で公開（要・あなたのStreamlitアカウント）

1. https://share.streamlit.io を開く → **GitHubでサインイン**（無料）
2. **「New app」** → リポジトリ `kyotei-yosou` を選択
3. 設定:
   - Branch: `main`
   - **Main file path: `streamlit_app.py`**
4. **Deploy** を押す → 数分でビルド完了
5. 固定URL `https://<好きな名前>.streamlit.app` が発行される（PC不要・常時起動）

---

## 3. 毎朝の自動更新（データを最新に保つ）

`run_daily.bat`（タスク `KyoteiPaperTrade`, 毎朝9:00）が:
1. 本日の妙味レースをスキャン → `data/today_picks.json` 更新
2. ペーパートレード精算・記録
3. **`git push`** で GitHub に反映 → Streamlit Cloud が自動で最新表示

※ push を効かせるには PC の git に認証情報の保存が必要:
```bash
git config --global credential.helper manager   # 一度認証すれば以降自動
```
（または方法Bのトークン入りremote URLにしておく）

---

## クラウドでできること / 制約
- ✅ **レース個別予想**: その場で boatrace.jp から取得して予測（PC不要・モデル同梱）
- ✅ **本日の妙味レース**: 毎朝pushされた `today_picks.json` を表示
- ⚠️ 無料枠はアクセスが無いとスリープ→次回アクセスで自動起動（数十秒）
- ⚠️ scikit-learn は 1.6.1 固定（モデルpickle互換のため。requirements.txt済）

---

## 取り消し
- ローカルタスク停止: `schtasks /delete /tn KyoteiPaperTrade /f`
- Streamlit Cloud: ダッシュボードからアプリ削除

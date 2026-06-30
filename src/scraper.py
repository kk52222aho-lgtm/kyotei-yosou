"""boatrace.jp 公式サイトのスクレイパー。

- 出走表 (racelist): 各艇の選手成績・モーター/ボート成績などの特徴量
- 結果 (raceresult): 着順（学習ターゲット）

公式サーバへの負荷を避けるため、リクエスト間に必ず sleep を入れる。
"""
from __future__ import annotations

import re
import time
import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BASE = "https://www.boatrace.jp/owpc/pc/race"
HEADERS = {"User-Agent": "Mozilla/5.0 (kyotei-yosou research; personal use)"}
SLEEP_SEC = 1.0  # マナーとしての待機

_session = requests.Session()
_session.headers.update(HEADERS)

# 全角数字 -> 半角
_Z2H = {ord(z): ord(h) for z, h in zip("０１２３４５６７８９", "0123456789")}


def _get(url: str, retries: int = 3) -> BeautifulSoup | None:
    for i in range(retries):
        try:
            r = _session.get(url, timeout=20)
            if r.status_code == 200:
                html = r.content.decode("utf-8", "replace")
                return BeautifulSoup(html, "html.parser")
        except requests.RequestException:
            pass
        time.sleep(SLEEP_SEC * (i + 1))
    return None


def _floats(text: str) -> list[float]:
    out = []
    for tok in re.findall(r"-?\d+\.?\d*", text):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _race_url(kind: str, date: str, jcd: str, rno: int) -> str:
    return f"{BASE}/{kind}?rno={rno}&jcd={str(jcd).zfill(2)}&hd={date}"


def fetch_racelist(date: str, jcd: str, rno: int) -> list[dict] | None:
    """出走表をパースして 6 艇分の特徴量 dict のリストを返す。

    date: 'YYYYMMDD', jcd: 場コード, rno: レース番号(1-12)
    レースが存在しない場合は None。
    """
    soup = _get(_race_url("racelist", date, jcd, rno))
    if soup is None:
        return None

    # 各選手が tbody 1個(4行) で表現される本体テーブルを探す
    target = None
    for t in soup.find_all("table"):
        if len(t.find_all("tbody")) >= 6:
            target = t
            break
    if target is None:
        return None

    entries: list[dict] = []
    for tb in target.find_all("tbody")[:6]:
        rows = tb.find_all("tr")
        if not rows:
            continue
        tds = rows[0].find_all("td")
        if len(tds) < 8:
            continue

        def cell(i: int) -> str:
            return re.sub(r"\s+", " ", tds[i].get_text(" ", strip=True)) if i < len(tds) else ""

        lane = _floats(cell(0))
        lane = int(lane[0]) if lane else len(entries) + 1

        # cell2: "4321 / A1 山田 太郎 1期/愛知 35歳/52.0kg"
        c2 = cell(2)
        m_reg = re.search(r"(\d{4})", c2)
        m_cls = re.search(r"\b(A1|A2|B1|B2)\b", c2)
        m_age = re.search(r"(\d+)歳", c2)
        m_wt = re.search(r"([\d.]+)kg", c2)
        # 名前: 級別の後ろ ～ 支部(○期/県 or 県) の前
        name = ""
        if m_cls:
            rest = c2[m_cls.end():].strip()
            rest = re.split(r"\d+期|/|\d+歳", rest)[0].strip()
            # "姓 名 出身地..." の先頭2トークンを選手名とする
            name = " ".join(rest.split()[:2])

        c3 = _floats(cell(3))  # F, L, 平均ST
        nat = _floats(cell(4))  # 全国 勝率/2連率/3連率
        loc = _floats(cell(5))  # 当地 勝率/2連率/3連率
        mot = _floats(cell(6))  # モーターNo/2連率/3連率
        boa = _floats(cell(7))  # ボートNo/2連率/3連率

        entries.append({
            "lane": lane,
            "reg": m_reg.group(1) if m_reg else "",
            "racer_class": m_cls.group(1) if m_cls else "",
            "name": name,
            "age": float(m_age.group(1)) if m_age else None,
            "weight": float(m_wt.group(1)) if m_wt else None,
            "f_count": int(c3[0]) if len(c3) >= 1 else 0,
            "l_count": int(c3[1]) if len(c3) >= 2 else 0,
            "avg_st": c3[2] if len(c3) >= 3 else None,
            "nat_win": nat[0] if len(nat) >= 1 else None,
            "nat_2rate": nat[1] if len(nat) >= 2 else None,
            "nat_3rate": nat[2] if len(nat) >= 3 else None,
            "loc_win": loc[0] if len(loc) >= 1 else None,
            "loc_2rate": loc[1] if len(loc) >= 2 else None,
            "loc_3rate": loc[2] if len(loc) >= 3 else None,
            "motor_2rate": mot[1] if len(mot) >= 2 else None,
            "motor_3rate": mot[2] if len(mot) >= 3 else None,
            "boat_2rate": boa[1] if len(boa) >= 2 else None,
            "boat_3rate": boa[2] if len(boa) >= 3 else None,
        })

    return entries if len(entries) == 6 else (entries or None)


def _st_to_float(text: str) -> float | None:
    """ST展示文字列をfloatに。'F.09'(フライング)は負値、'.05'は0.05。"""
    text = text.strip()
    if not text:
        return None
    flying = "F" in text.upper()
    m = re.search(r"\.?\d+", text)
    if not m:
        return None
    raw = m.group(0)
    val = float("0" + raw) if raw.startswith(".") else float(raw)
    if val >= 1:  # ".05" が "05" のように来た場合の保険
        val /= 100.0
    return -val if flying else val


def fetch_beforeinfo(date: str, jcd: str, rno: int) -> tuple[dict[int, dict], dict] | None:
    """直前情報をパース。

    返り値: (per_lane, race_cond)
      per_lane[艇番] = {tenji_time, tilt, weight_today, tenji_st}
      race_cond = {wind_speed, wave_height, temperature, water_temp}
    取得不可なら None。
    """
    soup = _get(_race_url("beforeinfo", date, jcd, rno))
    if soup is None:
        return None

    # 本体テーブル(tbody6) : 体重/展示タイム/チルト
    body = None
    for t in soup.find_all("table"):
        if len(t.find_all("tbody")) >= 6 and "展示" in t.get_text():
            body = t
            break

    per_lane: dict[int, dict] = {}
    if body is not None:
        for tb in body.find_all("tbody")[:6]:
            rows = tb.find_all("tr")
            if not rows:
                continue
            tds = [re.sub(r"\s+", " ", c.get_text(" ", strip=True))
                   for c in rows[0].find_all("td")]
            lane_v = _floats(tds[0]) if tds else []
            if not lane_v:
                continue
            lane = int(lane_v[0])
            wt = _floats(tds[3]) if len(tds) > 3 else []
            tt = _floats(tds[4]) if len(tds) > 4 else []
            tl = _floats(tds[5]) if len(tds) > 5 else []
            per_lane[lane] = {
                "weight_today": wt[0] if wt else None,
                "tenji_time": tt[0] if tt else None,
                "tilt": tl[0] if tl else None,
                "tenji_st": None,
            }

    # スタート展示テーブル: 各行 "枠 ST"
    for t in soup.find_all("table"):
        txt = t.get_text(" ", strip=True)
        if "スタート展示" in txt or ("ST" in txt and "コース" in txt):
            for tr in t.find_all("tr"):
                cell = re.sub(r"\s+", " ", tr.get_text(" ", strip=True))
                m = re.match(r"^([1-6])\s+(F?\.?\d+)", cell)
                if m:
                    lane = int(m.group(1))
                    st = _st_to_float(m.group(2))
                    if lane in per_lane:
                        per_lane[lane]["tenji_st"] = st
            break

    # 気象 (.weather1_bodyUnit のテキスト集合)
    race_cond = {"wind_speed": None, "wave_height": None,
                 "temperature": None, "water_temp": None}
    wtext = " ".join(u.get_text(" ", strip=True)
                     for u in soup.select(".weather1_bodyUnit"))
    m = re.search(r"風速\s*([\d.]+)\s*m", wtext)
    if m: race_cond["wind_speed"] = float(m.group(1))
    m = re.search(r"波高\s*([\d.]+)\s*cm", wtext)
    if m: race_cond["wave_height"] = float(m.group(1))
    m = re.search(r"気温\s*([\d.]+)", wtext)
    if m: race_cond["temperature"] = float(m.group(1))
    m = re.search(r"水温\s*([\d.]+)", wtext)
    if m: race_cond["water_temp"] = float(m.group(1))

    if not per_lane:
        return None
    return per_lane, race_cond


def fetch_odds(date: str, jcd: str, rno: int) -> dict[int, float] | None:
    """単勝オッズページ(oddstf)から {艇番: 単勝オッズ} を返す。

    未確定(0.0)・欠場などは値を入れない。締切前はライブのオッズ、
    過去レースは確定オッズが返る。
    """
    soup = _get(_race_url("oddstf", date, jcd, rno))
    if soup is None:
        return None

    target = None
    for t in soup.find_all("table"):
        if "単勝オッズ" in t.get_text():
            target = t
            break
    if target is None:
        return None

    odds: dict[int, float] = {}
    for tr in target.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        lane_raw = cells[0].get_text(strip=True)
        if not lane_raw.isdigit():
            continue
        lane = int(lane_raw)
        vals = _floats(cells[-1].get_text(" ", strip=True))
        if vals and vals[0] > 0:
            odds[lane] = vals[0]
    return odds or None


def fetch_held_venues(date: str) -> list[str]:
    """開催一覧(index)ページから、その日に開催している場コードを返す。"""
    soup = _get(f"{BASE}/index?hd={date}")
    if soup is None:
        return []
    jcds = set(re.findall(r"jcd=(\d{2})", str(soup)))
    return sorted(jcds)


def trifecta_combos() -> list[tuple[int, int, int]]:
    """3連単オッズ表(odds3t)の td.oddsPoint 出現順に対応する (1着,2着,3着) の列挙。

    DOM は行優先(20行×6列)。列 col(0-5) の1着=col+1、行 r は
    残り5艇を2着→3着で列挙した r 番目。
    """
    combos: list[tuple[int, int, int]] = [None] * 120  # type: ignore
    for col in range(6):
        first = col + 1
        others = [x for x in range(1, 7) if x != first]
        r = 0
        for second in others:
            for third in others:
                if third == second:
                    continue
                combos[r * 6 + col] = (first, second, third)
                r += 1
    return combos


_TRIFECTA_ORDER = trifecta_combos()


def fetch_trifecta_odds(date: str, jcd: str, rno: int) -> dict[str, float] | None:
    """3連単オッズページ(odds3t)から {'i-j-k': オッズ} を返す（最大120点）。"""
    soup = _get(_race_url("odds3t", date, jcd, rno))
    if soup is None:
        return None
    cells = soup.select("td.oddsPoint")
    if len(cells) < 120:
        return None
    out: dict[str, float] = {}
    for idx, td in enumerate(cells[:120]):
        combo = _TRIFECTA_ORDER[idx]
        vals = _floats(td.get_text(strip=True))
        if combo and vals and vals[0] > 0:
            out[f"{combo[0]}-{combo[1]}-{combo[2]}"] = vals[0]
    return out or None


def fetch_result(date: str, jcd: str, rno: int) -> dict[int, int] | None:
    """結果ページから {艇番: 着順} を返す。非完走(転覆/失格等)は着順 99。"""
    soup = _get(_race_url("raceresult", date, jcd, rno))
    if soup is None:
        return None

    # 着/枠/ボートレーサー/レースタイム を含むテーブルを探す
    target = None
    for t in soup.find_all("table"):
        head = t.get_text(" ", strip=True)[:30]
        if "着" in head and "ボートレーサー" in head:
            target = t
            break
    if target is None:
        return None

    finish: dict[int, int] = {}
    for tr in target.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        rank_raw = cells[0].get_text(strip=True).translate(_Z2H)
        lane_raw = cells[1].get_text(strip=True).translate(_Z2H)
        if not lane_raw.isdigit():
            continue
        lane = int(lane_raw)
        rank = int(rank_raw) if rank_raw.isdigit() else 99  # 転/失/欠 -> 99
        finish[lane] = rank

    return finish or None

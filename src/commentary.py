"""エージェント実況コメント生成（テンプレbase・LLM不使用＝無料/決定的で毎レース回せる）。

『全レースを直前まで予想→勝負or見送り→結果that出たら一言』の"一言"を作る。
入力: race(予想+買い目) と result(fetch_result_full出力 or None=未確定)。
※LLMを使わんのはクォータ制約(日150レース不可)。口調はエージェント風に固定テンプレ。
  将来リッチ化するなら勝負レース(数本)だけLLM、は別途可。
"""
from __future__ import annotations

import json
import os

from . import storage

AGENT_CACHE = os.path.join(storage.DATA_DIR, "today_agent.json")


def save_log(date: str, races: list[dict]) -> None:
    """本日エージェントが検討した全レース(締切順)を保存。"""
    os.makedirs(storage.DATA_DIR, exist_ok=True)
    ranked = sorted(races, key=lambda r: (r.get("deadline") or "99:99", r.get("venue", ""), r.get("rno", 0)))
    with open(AGENT_CACHE, "w", encoding="utf-8") as f:
        json.dump({"date": date, "races": ranked}, f, ensure_ascii=False, indent=1)


def load_log() -> dict | None:
    if not os.path.exists(AGENT_CACHE):
        return None
    with open(AGENT_CACHE, encoding="utf-8") as f:
        return json.load(f)


def _hits(r: dict, res: dict) -> dict:
    return {
        "single": res.get("winner") == r.get("honmei"),
        "exacta": res.get("exacta_combo") in (r.get("exacta3") or []),
        "trio": res.get("trio_combo") in (r.get("trio4") or []),
        "trifecta": res.get("trifecta_combo") in (r.get("trifecta3") or []),
    }


def comment(r: dict, res: dict | None) -> str:
    """race予想 r と 結果 res(=Noneなら未確定) から実況の一言。"""
    honmei = r.get("honmei")
    bet = r.get("bet")

    if not res:  # 未確定
        if bet:
            return (f"🔵 本命{honmei}号で勝負。単勝＋2連単3点＋3連複4点＋3連単3点、"
                    f"締切{r.get('deadline', '')}…結果待ち。")
        return f"⚪ 見送り。本命{honmei}号（{r.get('win_pct', 0):.0f}%）＝妙味薄。静観。"

    winner = res.get("winner")
    if bet:
        h = _hits(r, res)
        if h["single"]:
            extra = []
            if h["trifecta"]:
                extra.append(f"3連単{res.get('trifecta_yen')}円")
            elif h["exacta"]:
                extra.append(f"2連単{res.get('exacta_yen')}円")
            if h["trio"]:
                extra.append(f"3連複{res.get('trio_yen')}円")
            tail = ("＋" + "・".join(extra)) if extra else ""
            return (f"🎯 読み的中！本命{honmei}号that来た。単勝{res.get('tansho_yen')}円{tail}。"
                    f"インを否定した妙味that刺さった。")
        if winner == 1:
            if h["trio"]:
                return (f"😮 本命{honmei}号は届かず、インが残った…が3連複{res.get('trio_yen')}円で拾った。"
                        f"着順不問that効いた、これthaクッション。")
            return f"😖 本命{honmei}号届かず、イン(1号)that残った。荒れ読み外れ、この回は勉強代。"
        # 別の非イン艇that勝った
        if h["trio"]:
            return (f"🤔 本命{honmei}号でなく{winner}号。荒れ方向は合ったが本命違い。"
                    f"3連複{res.get('trio_yen')}円で救われた。")
        return f"😑 本命{honmei}号でなく{winner}号that来た。荒れたが本命外し、この回は取れず。"

    # 見送り・確定
    if winner == 1:
        return f"⚪ 見送り正解。インが固く1号that逃げた（単勝{res.get('tansho_yen')}円）。読み通り。"
    return f"👀 見送ったが{winner}号that来て荒れた。妙味と見なかった回、ルール通りとはいえ悔しい。"

"""FastAPI による予想表示。

  uvicorn src.app:app --reload
  → http://127.0.0.1:8000
"""
from __future__ import annotations

import datetime as dt
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import predict, scan
from .venues import VENUES, LOCAL_VENUES, name as venue_name

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI(title="競艇予想")


def today() -> str:
    return dt.date.today().strftime("%Y%m%d")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "venues": VENUES,
        "local": LOCAL_VENUES,
        "today": today(),
        "model_ready": predict.load_model() is not None,
    })


@app.get("/today", response_class=HTMLResponse)
def today_picks(request: Request):
    cache = scan.load_cache()
    return templates.TemplateResponse(request, "today.html", {
        "cache": cache,
        "venue_name": venue_name,
    })


@app.get("/race", response_class=HTMLResponse)
def race(request: Request, jcd: str, rno: int, hd: str = ""):
    hd = hd or today()
    error = None
    rows = None
    try:
        rows = predict.predict_race(hd, jcd, rno)
        if rows is None:
            error = "このレースの出走表が取得できませんでした（非開催 or 未発表）。"
    except RuntimeError as e:
        error = str(e)

    # 検証済み「勝てる形」の推奨買い目
    rec = predict.recommend(rows) if rows else None
    return templates.TemplateResponse(request, "race.html", {
        "jcd": jcd,
        "venue": venue_name(jcd),
        "rno": rno,
        "hd": hd,
        "rows": rows,
        "error": error,
        "rec": rec,
    })

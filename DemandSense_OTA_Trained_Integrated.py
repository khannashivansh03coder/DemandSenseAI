import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="DemandSense", page_icon="🏨", layout="wide",
                   initial_sidebar_state="collapsed")

BASE_DIR = Path(__file__).resolve().parent
FORECAST_PATH = BASE_DIR / "demandsense_30_day_forecast.csv"
HISTORY_CANDIDATES = [
    BASE_DIR / "demandsense_consistent.csv",
    BASE_DIR / "demandsense_cleaned.csv",
    BASE_DIR / "demandsense_synthetic_hotel_demand.csv",
]

# Internal level -> (display word, emoji, gradient colours)
LEVELS = {
    "High": ("Busy", "🔴", ("#e5484d", "#b3261e")),
    "Normal": ("Normal", "🟢", ("#2fa66a", "#1c7a4d")),
    "Low": ("Quiet", "🔵", ("#3b82d6", "#2456a6")),
}
CHART_DOMAIN = ["Busy", "Normal", "Quiet"]
CHART_RANGE = ["#e5484d", "#2fa66a", "#3b82d6"]

CHANNEL_COLS = ["Commission %", "Promo discount %", "Cancellation %",
                "Acquisition ₹/booking", "Daily campaign cost ₹", "Max extra bookings/day"]

# Illustrative defaults. Replace with the hotel's real channel terms.
CHANNEL_DEFAULTS = pd.DataFrame([
    {"Channel": "Direct Website", "Commission %": 0.0, "Promo discount %": 0.0, "Cancellation %": 8.0,
     "Acquisition ₹/booking": 200.0, "Daily campaign cost ₹": 1500.0, "Max extra bookings/day": 4},
    {"Channel": "MakeMyTrip", "Commission %": 18.0, "Promo discount %": 3.0, "Cancellation %": 18.0,
     "Acquisition ₹/booking": 0.0, "Daily campaign cost ₹": 0.0, "Max extra bookings/day": 8},
    {"Channel": "Goibibo", "Commission %": 18.0, "Promo discount %": 3.0, "Cancellation %": 17.0,
     "Acquisition ₹/booking": 0.0, "Daily campaign cost ₹": 0.0, "Max extra bookings/day": 6},
    {"Channel": "Booking.com", "Commission %": 17.0, "Promo discount %": 2.0, "Cancellation %": 20.0,
     "Acquisition ₹/booking": 0.0, "Daily campaign cost ₹": 0.0, "Max extra bookings/day": 6},
    {"Channel": "Agoda", "Commission %": 18.0, "Promo discount %": 2.0, "Cancellation %": 16.0,
     "Acquisition ₹/booking": 0.0, "Daily campaign cost ₹": 0.0, "Max extra bookings/day": 4},
    {"Channel": "Expedia", "Commission %": 18.0, "Promo discount %": 2.0, "Cancellation %": 19.0,
     "Acquisition ₹/booking": 0.0, "Daily campaign cost ₹": 0.0, "Max extra bookings/day": 3},
])

OTA_BOOKINGS_MODEL = BASE_DIR / "models" / "demandsense_ota_bookings_model.joblib"
OTA_CANCEL_MODEL = BASE_DIR / "models" / "demandsense_ota_cancellation_model.joblib"

def predict_ota_outcomes(date, occ_pct, room_rate, channel_row):
    booking_model = joblib.load(OTA_BOOKINGS_MODEL)
    cancel_model = joblib.load(OTA_CANCEL_MODEL)
    month = date.month
    season = "peak" if month in [4, 5, 6, 10, 11, 12] else ("shoulder" if month in [2, 3, 7, 9] else "low")
    row = {"channel": str(channel_row["Channel"]), "season": season, "day_of_week": int(date.dayofweek),
           "month": month, "is_weekend": int(date.dayofweek >= 5), "is_event": 0,
           "demand_index": float(np.clip(occ_pct, 0, 100)), "room_rate": float(room_rate),
           "discount_pct": float(channel_row["Promo discount %"]), "commission_pct": float(channel_row["Commission %"]), "lead_days": 7}
    X = pd.DataFrame([row])
    return max(0.0, float(booking_model.predict(X)[0])), float(cancel_model.predict_proba(X)[0, 1])

# -------------------- Styling --------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
.block-container {padding-top: 1.2rem; max-width: 1150px;}
.hero {background: linear-gradient(120deg,#2b1a4a 0%,#7a2f4f 45%,#e8683a 80%,#f5b53c 100%);
  border-radius: 20px; padding: 28px 34px; color: #fff; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2.2rem; font-weight: 800; color:#fff;}
.hero p {margin: 6px 0 0 0; font-size: 1.05rem; opacity: .92; color:#fff;}
.card {border-radius: 18px; padding: 20px 22px; color: #fff; min-height: 178px;
  box-shadow: 0 6px 18px rgba(0,0,0,.18);}
.card .t {font-size: .85rem; letter-spacing: .08em; text-transform: uppercase; opacity: .85;}
.card .big {font-size: 3.1rem; font-weight: 800; line-height: 1.1; margin: 4px 0;}
.card .badge {display:inline-block; background: rgba(255,255,255,.22); border-radius: 999px;
  padding: 3px 12px; font-size: .85rem; font-weight: 600;}
.card .n {font-size: .92rem; margin-top: 10px; opacity: .95;}
.pick {border-radius: 16px; padding: 16px 18px; background: #fff7e6; border: 1px solid #f5d9a0; color:#3a2a10; min-height: 128px;}
.pick.best {background: #fff1c9; border: 2px solid #f5b53c;}
.pick.weak {background: #fdecec; border: 1px solid #f3b9b9;}
.pick .h {font-size: .8rem; text-transform: uppercase; letter-spacing: .07em; opacity: .75;}
.pick .name {font-size: 1.45rem; font-weight: 800; margin: 2px 0;}
.pick .v {font-size: .95rem;}
.advice {border-radius: 14px; padding: 14px 18px; background: rgba(120,120,160,.12); font-size: 1.02rem;}
</style>
""", unsafe_allow_html=True)


# -------------------- Helpers --------------------
def inr(x):
    return f"₹{x:,.0f}"


def to_num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def demand_level(occ, low_thr, high_thr):
    if occ >= high_thr:
        return "High"
    if occ < low_thr:
        return "Low"
    return "Normal"


def read_forecast(df):
    """Accepts columns date, predicted_occupancy (+ optional total_rooms, avg_price)."""
    need = {"date", "predicted_occupancy"}
    if not need.issubset(df.columns):
        return None, None, None
    f = df.copy()
    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.normalize()
    f["predicted_occupancy"] = pd.to_numeric(f["predicted_occupancy"], errors="coerce")
    f = f.dropna(subset=["date", "predicted_occupancy"])
    if f.empty:
        return None, None, None
    if f["predicted_occupancy"].max() <= 1.5:
        f["predicted_occupancy"] = f["predicted_occupancy"] * 100
    f["predicted_occupancy"] = f["predicted_occupancy"].clip(0, 100)
    rooms = None
    if "total_rooms" in f.columns:
        r = pd.to_numeric(f["total_rooms"], errors="coerce").dropna()
        rooms = int(r.mode().iloc[0]) if not r.empty else None
    rate = None
    if "avg_price" in f.columns:
        p = pd.to_numeric(f["avg_price"], errors="coerce").dropna()
        rate = float(p.median()) if not p.empty else None
    return dict(zip(f["date"], f["predicted_occupancy"])), rooms, rate


def read_history(df):
    """Best-effort detection of a date column and an occupancy column in a history file."""
    cols = {c.lower(): c for c in df.columns}
    date_col = next((cols[k] for k in cols if k == "date"), None) or \
        next((cols[k] for k in cols if "date" in k), None)
    if date_col is None:
        return None, None, None
    occ = None
    cands = [orig for low, orig in cols.items()
             if "occup" in low and "room" not in low and "vacan" not in low and "predicted" not in low]
    if cands:
        occ = to_num(df[cands[0]], np.nan)
        if occ.max() <= 1.5:
            occ = occ * 100
    else:
        occupied = next((orig for low, orig in cols.items() if "occupied" in low and "predicted" not in low), None)
        total = next((orig for low, orig in cols.items() if "total_rooms" in low or low == "rooms"), None)
        if occupied and total:
            occ = to_num(df[occupied], np.nan) / to_num(df[total], np.nan).replace(0, np.nan) * 100
    if occ is None:
        return None, None, None
    h = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(), "occ": occ})
    h = h.dropna()
    if h.empty:
        return None, None, None
    h["occ"] = h["occ"].clip(0, 100)
    rooms = None
    tcol = next((orig for low, orig in cols.items() if "total_rooms" in low), None)
    if tcol:
        r = pd.to_numeric(df[tcol], errors="coerce").dropna()
        rooms = int(r.mode().iloc[0]) if not r.empty else None
    rate = None
    if "avg_price" in df.columns:
        p = pd.to_numeric(df["avg_price"], errors="coerce").dropna()
        rate = float(p.median()) if not p.empty else None
    return h, rooms, rate


def make_profile_predictor(hist):
    """Seasonal estimate: average occupancy for the same month and weekday in the history."""
    h = hist.copy()
    h["m"], h["d"] = h["date"].dt.month, h["date"].dt.dayofweek
    md = h.groupby(["m", "d"])["occ"].agg(["mean", "count"])
    dd = h.groupby("d")["occ"].mean()
    overall = float(h["occ"].mean())

    def predict(date):
        key = (date.month, date.dayofweek)
        if key in md.index and md.loc[key, "count"] >= 2:
            return float(md.loc[key, "mean"])
        if date.dayofweek in dd.index:
            return float(dd.loc[date.dayofweek])
        return overall
    return predict


def clean_channels(df):
    d = df.copy()
    d["Channel"] = d["Channel"].astype(str)
    for col in CHANNEL_COLS:
        d[col] = to_num(d[col]).clip(lower=0)
    d["Commission %"] = d["Commission %"].clip(upper=100)
    d["Promo discount %"] = d["Promo discount %"].clip(upper=100)
    d["Cancellation %"] = d["Cancellation %"].clip(upper=95)
    return d


def channel_economics(ch, adr, service_cost):
    out = ch.copy()
    stay_share = 1 - out["Cancellation %"] / 100
    gross = adr * (1 - out["Promo discount %"] / 100)
    commission = gross * out["Commission %"] / 100
    acq_per_stay = out["Acquisition ₹/booking"] / stay_share
    out["Commission ₹"] = commission
    out["Net per stayed room ₹"] = gross - commission - service_cost - acq_per_stay
    out["Max stays/day"] = out["Max extra bookings/day"] * stay_share
    return out.sort_values("Net per stayed room ₹", ascending=False).reset_index(drop=True)


def allocate(econ, rooms_to_fill):
    remaining = float(rooms_to_fill)
    rows = []
    for _, r in econ.iterrows():
        net = float(r["Net per stayed room ₹"])
        campaign_cost = float(r["Daily campaign cost ₹"])
        stays = 0.0
        if remaining > 1e-9 and net > 0:
            stays = min(remaining, float(r["Max stays/day"]))
            if stays * net - campaign_cost <= 0:
                stays = 0.0
        campaign = campaign_cost if stays > 0 else 0.0
        remaining -= stays
        rows.append({"Channel": r["Channel"], "stays": stays,
                     "net": stays * net - campaign, "per_room": net})
    return pd.DataFrame(rows), max(0.0, remaining)


def baseline_spread(econ, rooms_to_fill):
    cap = float(econ["Max stays/day"].sum())
    filled = min(float(rooms_to_fill), cap)
    if cap <= 0 or filled <= 0:
        return 0.0
    stays = econ["Max stays/day"] / cap * filled
    net = float((stays * econ["Net per stayed room ₹"]).sum())
    return net - float(econ.loc[stays > 0, "Daily campaign cost ₹"].sum())


def is_direct(name):
    return "direct" in str(name).lower()


def draw(fn, fallback):
    try:
        chart = fn()
        try:
            st.altair_chart(chart, width="stretch")
        except TypeError:
            st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(fallback)


def card(title, big, level, note):
    word, _, (c1, c2) = LEVELS[level]
    st.markdown(
        f"""<div class="card" style="background:linear-gradient(135deg,{c1},{c2})">
        <div class="t">{title}</div><div class="big">{big}</div>
        <span class="badge">{word} demand</span><div class="n">{note}</div></div>""",
        unsafe_allow_html=True)


# -------------------- Sidebar: advanced settings --------------------
with st.sidebar:
    st.header("⚙ Advanced settings")
    st.caption("Most people never need this. Defaults work for the demo.")
    up = st.file_uploader("Use my own forecast CSV (optional)", type="csv",
                          help="Columns: date, predicted_occupancy. Optional: total_rooms, avg_price")
    # Defaults better match the range of the current trained 30-day forecast.
    # The user can still adjust these thresholds for their hotel's own definition of quiet/busy.
    low_thr = st.number_input("Quiet day if occupancy below (%)", 0.0, 100.0, 55.0, 1.0)
    high_thr = st.number_input("Busy day if occupancy from (%)", 0.0, 100.0, 68.0, 1.0)
    target_occ = st.slider("On quiet days, aim to reach (% occupancy)", 40, 90, 60)
    service_cost = st.number_input("Cost to serve one guest (₹)", 0.0, 10000.0, 700.0, 50.0)
    st.markdown("**Booking site terms** (edit to match your hotel)")
    edited = st.data_editor(
        CHANNEL_DEFAULTS, hide_index=True, num_rows="fixed", disabled=["Channel"], key="channels",
        column_config={
            "Commission %": st.column_config.NumberColumn(min_value=0, max_value=100, step=0.5, format="%.1f"),
            "Promo discount %": st.column_config.NumberColumn(min_value=0, max_value=100, step=0.5, format="%.1f"),
            "Cancellation %": st.column_config.NumberColumn(min_value=0, max_value=95, step=1, format="%.1f"),
            "Acquisition ₹/booking": st.column_config.NumberColumn(min_value=0, step=50, format="₹%.0f"),
            "Daily campaign cost ₹": st.column_config.NumberColumn(min_value=0, step=250, format="₹%.0f"),
            "Max extra bookings/day": st.column_config.NumberColumn(min_value=0, step=1, format="%d"),
        })
channels = clean_channels(edited)

# -------------------- Load data --------------------
fc_map, fc_rooms, fc_rate = {}, None, None
try:
    if up is not None:
        fc_map, fc_rooms, fc_rate = read_forecast(pd.read_csv(up))
    elif FORECAST_PATH.exists():
        fc_map, fc_rooms, fc_rate = read_forecast(pd.read_csv(FORECAST_PATH))
except Exception:
    fc_map = None
fc_map = fc_map or {}

hist, h_rooms, h_rate = None, None, None
for p in HISTORY_CANDIDATES:
    if p.exists():
        try:
            hist, h_rooms, h_rate = read_history(pd.read_csv(p))
        except Exception:
            hist = None
        if hist is not None:
            break

# Use the project's historical dataset for seasonal estimates outside the trained
# 30-day forecast window. Never silently substitute synthetic history.
if hist is None:
    st.error(
        "Historical data was not found. Keep demandsense_consistent.csv "
        "(or demandsense_cleaned.csv) in the same folder as this app.py."
    )
    st.stop()

predict = make_profile_predictor(hist)
default_rooms = int(fc_rooms or h_rooms or 30)
default_rate = float(round((fc_rate or h_rate or 4000.0) / 100) * 100)
default_date = (min(fc_map) - pd.Timedelta(days=1)).date() if fc_map else dt.date.today()

# -------------------- Header and inputs --------------------
st.markdown("""<div class="hero"><h1>🏨 DemandSense</h1>
<p>Know how full your hotel will be, and which booking site makes you the most money.</p></div>""",
            unsafe_allow_html=True)

i1, i2, i3 = st.columns([2, 1, 1])
selected = i1.date_input("📅 Pick a date (forecast starts the day after)", value=default_date)
rooms = int(i2.number_input("🛏 Rooms in hotel", 1, 500, default_rooms, 1))
rate = float(i3.number_input("💰 Room rate (₹/night)", 500.0, 50000.0, default_rate, 100.0))

# -------------------- Build the 30-day plan --------------------
econ = channel_economics(channels, rate, service_cost)
start = pd.Timestamp(selected).normalize() + pd.Timedelta(days=1)
rows = []
for d in pd.date_range(start, periods=30):
    if d in fc_map:
        occ, src = fc_map[d], "Forecast model"
    else:
        occ, src = predict(d), "Seasonal estimate"
    level = demand_level(occ, low_thr, high_thr)
    occupied = int(round(rooms * occ / 100))
    vacant = max(0, rooms - occupied)
    to_fill, profit, gain, push = 0, 0.0, 0.0, []
    if level == "Low":
        target_rooms = int(np.ceil(rooms * target_occ / 100))
        to_fill = int(max(0, min(vacant, target_rooms - occupied)))
        alloc, _ = allocate(econ, to_fill)
        used = alloc[alloc["stays"] > 0]
        push = used["Channel"].tolist()[:3]
        profit = float(alloc["net"].sum())
        gain = profit - baseline_spread(econ, to_fill)
    rows.append({"date": d, "occ": float(occ), "level": level, "vacant": vacant, "to_fill": to_fill,
                 "push": push, "profit": profit, "gain": gain, "src": src})
plan = pd.DataFrame(rows)
plan["word"] = plan["level"].map(lambda l: LEVELS[l][0])
plan["Demand"] = plan["level"].map(lambda l: f"{LEVELS[l][1]} {LEVELS[l][0]}")

worst_otas = [c for c in econ["Channel"].tolist() if not is_direct(c)][-2:]


def advice(r):
    if r["level"] == "High":
        w = " and ".join(worst_otas) if worst_otas else "the weakest booking sites"
        return f"You will fill up anyway. Keep rates high, skip discounts, and limit {w} so you don't pay commission on rooms that would sell regardless."
    if r["level"] == "Low":
        if not r["push"]:
            return "Quiet day, but no booking site earns back its cost at these settings. Don't run a promotion."
        return (f"Quiet day. About {r['to_fill']} rooms can be filled. Push {', '.join(r['push'])} first. "
                f"Expected profit from those rooms: {inr(r['profit'])}.")
    return "Normal demand. Keep your usual rates, roster and orders."


def summary_card(title, sub):
    avg = float(sub["occ"].mean())
    lvl = demand_level(avg, low_thr, high_thr)
    busy, quiet = int((sub["level"] == "High").sum()), int((sub["level"] == "Low").sum())
    card(title, f"{avg:.0f}%", lvl, f"{busy} busy · {quiet} quiet day{'s' if quiet != 1 else ''}")


# -------------------- 3 summary cards --------------------
st.markdown("### How full will you be?")
tomorrow = plan.iloc[0]
c1, c2, c3 = st.columns(3)
with c1:
    card(f"Tomorrow · {tomorrow['date']:%a %d %b}", f"{tomorrow['occ']:.0f}%", tomorrow["level"],
         f"{tomorrow['vacant']} of {rooms} rooms likely empty")
with c2:
    summary_card("Next 7 days (average)", plan.iloc[:7])
with c3:
    summary_card("Next 30 days (average)", plan)
n_est = int((plan["src"] == "Seasonal estimate").sum())
if n_est:
    st.caption(
        f"{30 - n_est} day(s) use the trained forecast CSV and {n_est} day(s) use a seasonal estimate "
        f"from your historical dataset (same month and weekday). The seasonal estimate is not a fresh model prediction."
    )
else:
    st.caption("All 30 days shown are from the existing trained forecast CSV.")

# -------------------- Which booking site --------------------
st.markdown("### Which booking site should I use?")
comparison_date = pd.Timestamp(selected).normalize() + pd.Timedelta(days=1)
comparison_occ = float(plan.iloc[0]["occ"])
if OTA_BOOKINGS_MODEL.exists() and OTA_CANCEL_MODEL.exists():
    st.caption("Synthetic-demo OTA models: predictions illustrate the pipeline and are not real-world OTA performance claims.")
    result_rows = []
    for _, ch in channels.iterrows():
        pred_bookings, cancel_p = predict_ota_outcomes(comparison_date, comparison_occ, rate, ch)
        stays = pred_bookings * (1 - cancel_p)
        discounted_rate = rate * (1 - float(ch["Promo discount %"]) / 100)
        commission_per_stay = discounted_rate * float(ch["Commission %"]) / 100
        acquisition = float(ch["Acquisition ₹/booking"])
        campaign = float(ch["Daily campaign cost ₹"]) if pred_bookings >= 1 else 0.0
        gross = stays * discounted_rate
        commission = stays * commission_per_stay
        acquisition_cost = pred_bookings * acquisition
        service = stays * service_cost
        net = gross - commission - acquisition_cost - service - campaign
        result_rows.append({"Channel": ch["Channel"], "Predicted bookings": pred_bookings,
            "Cancellation probability %": cancel_p*100, "Expected stays": stays, "Gross revenue ₹": gross,
            "Commission ₹": commission, "Acquisition cost ₹": acquisition_cost, "Service cost ₹": service,
            "Campaign cost ₹": campaign, "Estimated net profit ₹": net})
    ota_results = pd.DataFrame(result_rows).sort_values("Estimated net profit ₹", ascending=False).reset_index(drop=True)
    if not ota_results.empty:
        best = ota_results.iloc[0]
        ota_only = ota_results[~ota_results["Channel"].map(is_direct)]
        best_ota = ota_only.iloc[0] if not ota_only.empty else best
        weakest = ota_results.iloc[-1]
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"<div class='pick best'><div class='h'>🥇 Best estimated channel</div><div class='name'>{best['Channel']}</div><div class='v'>Estimated net profit <b>{inr(best['Estimated net profit ₹'])}</b></div></div>", unsafe_allow_html=True)
        with p2:
            st.markdown(f"<div class='pick'><div class='h'>🏆 Best estimated OTA</div><div class='name'>{best_ota['Channel']}</div><div class='v'>Estimated net profit <b>{inr(best_ota['Estimated net profit ₹'])}</b></div></div>", unsafe_allow_html=True)
        with p3:
            st.markdown(f"<div class='pick weak'><div class='h'>⚠ Lowest estimated profit</div><div class='name'>{weakest['Channel']}</div><div class='v'>Estimated net profit <b>{inr(weakest['Estimated net profit ₹'])}</b></div></div>", unsafe_allow_html=True)
        st.markdown(f"**Date:** {comparison_date:%d %b %Y} · **Forecast occupancy:** {comparison_occ:.1f}% · **Hotel capacity:** {rooms} rooms")
        st.markdown(f"**Recommendation:** prioritise **{best['Channel']}** under the current synthetic demo assumptions. Estimated bookings: {best['Predicted bookings']:.1f}; estimated cancellation probability: {best['Cancellation probability %']:.1f}%. Replace synthetic assumptions with real channel data before using this for actual business decisions.")
        st.dataframe(ota_results.style.format({"Predicted bookings":"{:.1f}","Cancellation probability %":"{:.1f}%","Expected stays":"{:.1f}","Gross revenue ₹":"₹{:,.0f}","Commission ₹":"₹{:,.0f}","Acquisition cost ₹":"₹{:,.0f}","Service cost ₹":"₹{:,.0f}","Campaign cost ₹":"₹{:,.0f}","Estimated net profit ₹":"₹{:,.0f}"}), hide_index=True, use_container_width=True)
        st.bar_chart(ota_results.set_index("Channel")["Estimated net profit ₹"])
else:
    st.warning("OTA models are missing. Run build_ota_model.py first, then restart this dashboard.")

# -------------------- Plan tabs --------------------
st.markdown("### Your plan")
tab1, tab2, tab3 = st.tabs(["Tomorrow", "Next 7 days", "Next 30 days"])


def occ_chart(sub, short):
    import altair as alt
    d = sub[["date", "occ", "word"]].copy()
    d["label"] = d["date"].dt.strftime("%a %d" if short else "%a %d %b")
    d["Demand"] = d["word"]
    bars = alt.Chart(d).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("label:N", sort=None, title=None, axis=alt.Axis(labelAngle=-60 if short else 0)),
        y=alt.Y("occ:Q", title="Occupancy (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Demand:N", scale=alt.Scale(domain=CHART_DOMAIN, range=CHART_RANGE), legend=alt.Legend(orient="top")),
        tooltip=["label", alt.Tooltip("occ:Q", format=".0f", title="Occupancy %"), "Demand"])
    rules = alt.Chart(pd.DataFrame({"y": [low_thr, high_thr]})).mark_rule(strokeDash=[4, 4], color="#999").encode(y="y:Q")
    return (bars + rules).properties(height=300)


def day_table(sub):
    t = pd.DataFrame({
        "Date": sub["date"].dt.strftime("%a %d %b"),
        "Occupancy": sub["occ"].round(0),
        "Demand": sub["Demand"],
        "What to do": sub.apply(advice, axis=1),
    })
    st.dataframe(t, hide_index=True, column_config={
        "Occupancy": st.column_config.ProgressColumn("Occupancy", min_value=0, max_value=100, format="%d%%"),
        "What to do": st.column_config.TextColumn("What to do", width="large"),
    })


def money_line(sub):
    q = sub[sub["level"] == "Low"]
    if q.empty:
        st.info("No quiet days in this period, so no promotion is needed.")
        return
    m1, m2, m3 = st.columns(3)
    m1.metric("Quiet days", len(q))
    m2.metric("Rooms to fill", int(q["to_fill"].sum()))
    m3.metric("Expected profit from those rooms", inr(q["profit"].sum()),
              f"{inr(q['gain'].sum())} more than spreading evenly", delta_color="off")


with tab1:
    a, b = st.columns([1, 1])
    with a:
        st.markdown(f"**{tomorrow['date']:%A, %d %B}**")
        st.progress(min(1.0, tomorrow["occ"] / 100), text=f"{tomorrow['occ']:.0f}% full")
        x1, x2 = st.columns(2)
        x1.metric("Rooms likely booked", rooms - tomorrow["vacant"])
        x2.metric("Rooms likely empty", tomorrow["vacant"])
    with b:
        st.markdown(f"""<div class="advice">{advice(tomorrow)}</div>""", unsafe_allow_html=True)

with tab2:
    s7 = plan.iloc[:7]
    draw(lambda: occ_chart(s7, False), s7.set_index("date")["occ"])
    money_line(s7)
    day_table(s7)

with tab3:
    draw(lambda: occ_chart(plan, True), plan.set_index("date")["occ"])
    money_line(plan)
    day_table(plan)

st.caption("Prototype. Commission, discount and cancellation numbers are editable assumptions (⚙ in the sidebar), "
           "not measured results. Replace them with your hotel's real terms before acting on the money figures.")

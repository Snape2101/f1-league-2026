import json
import re
import unicodedata
from datetime import datetime, timezone

import gspread
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ==========================================
# 0. SETTINGS  (everything you may want to tweak lives here)
# ==========================================
SHEET_NAME = "Spádómskeppni (Responses)"
WORKSHEET = "Form Responses 1"
API = "https://api.jolpi.ca/ergast/f1"

# Timezone of the Google Sheet's "Timestamp" column (Sheet > File > Settings).
SHEET_TIMEZONE = "Atlantic/Reykjavik"
# Format of the Timestamp column as the Sheet shows it, e.g. "3/5/2026 9:04:58" = 5 March.
# (Month first. Change to "%d/%m/%Y %H:%M:%S" if your Sheet is ever switched to day first.)
TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"

# Scoring
EXACT_HIT_SCORE = -4      # score for a perfect position ("Pow")
NOT_CLASSIFIED_POS = 22   # position given to a guessed driver who isn't in the results

# Penalty for not submitting (see sidebar "Reglur"):
# a player who misses a race gets the WORST score submitted for that race x MISSED_MULTIPLIER
# (+ an optional flat MISSED_EXTRA), rounded to a whole number.
MISSED_MULTIPLIER = 1.38
MISSED_EXTRA = 0
# True  = penalties only count from a player's first valid entry (fair to late joiners)
# False = everyone is penalised for every race since the season started
PENALTY_FROM_FIRST_ENTRY = True

# True  = guesses submitted after the race start are ignored
# False = every guess counts, whenever it was submitted (use this if guesses were
#         entered retroactively, or to test whether the deadline is dropping races)
ENFORCE_DEADLINE = True

st.set_page_config(
    page_title="F1 Spádómskeppnin 2026",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2.5rem; font-weight: 800; color: #FF1801;
                   text-align: center; margin-bottom: 0px; }
    .sub-header  { font-size: 1.1rem; text-align: center; color: #888;
                   margin-bottom: 30px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 1. NAME / TRACK NORMALISATION
# ==========================================
DRIVER_ALIASES = {
    "kimi antonelli": "andrea kimi antonelli",
    "alex albon": "alexander albon",
    "carlos sainz jr": "carlos sainz",
    "carlos sainz jr.": "carlos sainz",
}

# Icelandic form value (normalised, no accents) -> Jolpica circuitId
TRACK_ALIASES = {
    "astralia": "albert_park",
    "kina": "shanghai",
    "japan": "suzuka",
    "barein": "sepang",       # NOTE: check this is really the circuitId the API returns
    "bahrain": "sepang",
    "bahrein": "sepang",
    "saudi arabia": "jeddah",
    "sadi arabia": "jeddah",
    "miami": "miami",
    "kanada": "villeneuve",
    "monako": "monaco",
    "monaco": "monaco",
    "spann": "catalunya",
    "austurriki": "red_bull_ring",
    "bretland": "silverstone",
    "silverstone": "silverstone",
    "belgia": "spa",
    "spa": "spa",
    "ungverjaland": "hungaroring",
    "budapest": "hungaroring",
    "holland": "zandvoort",
    "zandvoort": "zandvoort",
    "itali": "monza",
    "italia": "monza",
    "monza": "monza",
    "madrid": "madring",
    "aserbaidsjan": "baku",
    "baku": "baku",
    "singapur": "marina_bay",
    "bandarikin": "americas",
    "texas (cota)": "americas",
    "mexiko": "rodriguez",
    "brasilia": "interlagos",
    "interlagos": "interlagos",
    "las vegas": "vegas",
    "katars": "losail",
    "katar": "losail",
    "abu dhabi": "yas_marina",
    # labels that may appear in the Sheet next to the Icelandic name
    "albert park": "albert_park",
    "shanghai": "shanghai",
    "barcelona": "catalunya",
    "catalunya": "catalunya",
    "madring": "madring",
}
# Longest keys first so e.g. "spann" wins over "spa" in the substring fallback.
_ALIAS_KEYS_BY_LENGTH = sorted(TRACK_ALIASES, key=len, reverse=True)


def normalize_string(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")


def normalize_driver_name(name):
    name = normalize_string(name)
    return DRIVER_ALIASES.get(name, name)


def normalize_track(track_name):
    t = normalize_string(track_name)
    if t in TRACK_ALIASES:                 # exact match first
        return TRACK_ALIASES[t]
    for key in _ALIAS_KEYS_BY_LENGTH:      # tolerant fallback, e.g. "Ítalía (Monza)"
        if key in t:
            return TRACK_ALIASES[key]
    return t


# ==========================================
# 2. DATA LOADING (each source cached on its own)
# ==========================================
def race_start(race):
    t = race.get("time", "12:00:00Z")
    return f"{race['date']}T{t if t.endswith('Z') else t + 'Z'}"


@st.cache_data(ttl=300, show_spinner=False)
def load_guesses():
    creds = json.loads(st.secrets["GCP_CREDENTIALS_JSON"])
    gc = gspread.service_account_from_dict(creds)
    ws = gc.open(SHEET_NAME).worksheet(WORKSHEET)
    return pd.DataFrame(ws.get_all_records())


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_races():
    """All completed races of the season. Paginates, because the API caps page size."""
    merged, offset = {}, 0
    while True:
        res = requests.get(
            f"{API}/current/results.json",
            params={"limit": 100, "offset": offset},
            timeout=15,
        )
        res.raise_for_status()
        mr = res.json()["MRData"]
        for race in mr["RaceTable"]["Races"]:
            rnd = int(race["round"])
            entry = merged.setdefault(rnd, {
                "round": rnd,
                "name": race["raceName"],
                "circuit": race["Circuit"]["circuitId"],
                "circuit_name": race["Circuit"].get("circuitName", ""),
                "locality": race["Circuit"].get("Location", {}).get("locality", ""),
                "start": race_start(race),
                "results": [],
            })
            for d in race["Results"]:
                entry["results"].append({
                    "driver": f"{d['Driver']['givenName']} {d['Driver']['familyName']}",
                    "pos": int(d["position"]),
                })
        offset += int(mr["limit"])
        if offset >= int(mr["total"]):
            break
    return [merged[k] for k in sorted(merged)], datetime.now(timezone.utc)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_driver_standings():
    """Top 3 of the drivers' championship, or None if the API can't be reached."""
    try:
        res = requests.get(f"{API}/current/driverstandings.json", timeout=10)
        res.raise_for_status()
        lists = res.json()["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            return None
        return [{
            "pos": int(s["position"]),
            "driver": f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
            "points": float(s["points"]),
        } for s in lists[0]["DriverStandings"][:3]]
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_next_race():
    try:
        res = requests.get(f"{API}/current/next.json", timeout=10)
        res.raise_for_status()
        race = res.json()["MRData"]["RaceTable"]["Races"][0]
        return {"name": race["raceName"], "circuit": race["Circuit"]["circuitId"],
                "start": race_start(race)}
    except Exception:
        return None


# ==========================================
# 3. SCORING
# ==========================================
def compute(raw, races):
    df_races = pd.DataFrame([{
        "Match_Track": r["circuit"],
        "Round": r["round"],
        "Race": r["name"].replace(" Grand Prix", " GP"),
        "Start": pd.Timestamp(r["start"]),
        "Brautarnafn": r["circuit_name"],
    } for r in races])

    df_actual = pd.DataFrame([{
        "Match_Track": r["circuit"],
        "Match_Name": normalize_driver_name(d["driver"]),
        "Actual_Driver": d["driver"],
        "Actual_Pos": d["pos"],
    } for r in races for d in r["results"]])

    # --- resolve track ids against what the API actually calls its circuits ---
    api_text = {
        r["circuit"]: normalize_string(" ".join(
            [r["circuit"].replace("_", " "), r["circuit_name"], r["locality"], r["name"]]))
        for r in races
    }

    def resolve_track(track_id):
        """If the id isn't one the API uses, look for it in the API's circuit/race names."""
        if track_id in api_text:
            return track_id
        key = track_id.replace("_", " ")
        for cid, text in api_text.items():
            if re.search(rf"\b{re.escape(key)}\b", text):
                return cid
        return track_id

    # --- clean guesses ---
    g = raw.copy()
    g["Nafn"] = g["Nafn"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    g = g[(g["Nafn"] != "") & (g["Braut"].astype(str).str.strip() != "")].copy()
    g["Match_Track"] = g["Braut"].apply(lambda b: resolve_track(normalize_track(b)))

    ts = pd.to_datetime(g["Timestamp"].astype(str).str.strip(),
                        format=TIMESTAMP_FORMAT, errors="coerce")
    if ts.isna().any():
        st.sidebar.warning(
            f"{int(ts.isna().sum())} tímastimplar þekktust ekki — athugaðu TIMESTAMP_FORMAT.")
    g["Submitted"] = (
        ts.dt.tz_localize(SHEET_TIMEZONE, ambiguous="NaT", nonexistent="shift_forward")
          .dt.tz_convert("UTC")
    )

    players = sorted(g["Nafn"].unique())
    known = set(TRACK_ALIASES.values())
    unknown_tracks = sorted(g.loc[~g["Match_Track"].isin(known), "Braut"].astype(str).unique())

    # --- deadline: guesses submitted after the race start don't count ---
    g = g.merge(df_races[["Match_Track", "Start"]], on="Match_Track", how="left")
    g["Late"] = (g["Submitted"] > g["Start"]) & ENFORCE_DEADLINE
    on_time = g[~g["Late"]]
    entries_by_track = on_time.groupby("Match_Track")["Nafn"].apply(lambda s: sorted(set(s))).to_dict()

    # --- latest valid guess per player per completed race ---
    valid = (
        on_time[on_time["Start"].notna()]
        .sort_values("Submitted")
        .drop_duplicates(subset=["Nafn", "Match_Track"], keep="last")
    )

    cols = [f"{i}. sæti" for i in range(1, 9)]
    m = valid.melt(id_vars=["Nafn", "Match_Track"], value_vars=cols,
                   var_name="pos_text", value_name="Guess")
    m["Predicted_Pos"] = m["pos_text"].str.extract(r"(\d+)")[0].astype(int)
    m["Match_Name"] = m["Guess"].apply(normalize_driver_name)
    m = m.merge(df_actual, on=["Match_Track", "Match_Name"], how="left")
    m["Actual_Pos"] = m["Actual_Pos"].fillna(NOT_CLASSIFIED_POS)
    m["Difference"] = (m["Predicted_Pos"] - m["Actual_Pos"]).abs()
    m["Score"] = np.where(m["Difference"] == 0, EXACT_HIT_SCORE, m["Difference"])
    m["Pow"] = (m["Difference"] == 0).astype(int)

    scored = m.groupby(["Match_Track", "Nafn"]).agg(
        Score=("Score", "sum"), Pows=("Pow", "sum")).reset_index()

    # --- full player x race grid, then apply the missed-race penalty ---
    grid = pd.MultiIndex.from_product(
        [df_races["Match_Track"], players], names=["Match_Track", "Nafn"]
    ).to_frame(index=False)
    full = grid.merge(scored, on=["Match_Track", "Nafn"], how="left")
    full = full.merge(df_races[["Match_Track", "Round", "Race"]], on="Match_Track")
    full["Entered"] = full["Score"].notna()

    # ignore races where nobody entered at all
    full = full[full.groupby("Match_Track")["Entered"].transform("any")].copy()

    worst = full.groupby("Match_Track")["Score"].transform("max")
    missed = ~full["Entered"]
    if PENALTY_FROM_FIRST_ENTRY:
        first_round = full[full["Entered"]].groupby("Nafn")["Round"].min()
        missed &= full["Round"] >= full["Nafn"].map(first_round)

    penalty = (worst.clip(lower=0) * MISSED_MULTIPLIER + MISSED_EXTRA).round()
    full.loc[missed, "Score"] = penalty[missed]
    full.loc[missed, "Pows"] = 0
    full = full[full["Entered"] | missed].copy()
    full["Pows"] = full["Pows"].astype(int)
    full["Status"] = np.where(full["Entered"], "Skilaði", "Missti af")
    full = full.sort_values(["Round", "Nafn"]).reset_index(drop=True)

    # --- diagnostics: how do sheet rows line up with API races? ---
    counts = g.groupby("Match_Track").agg(
        Spár=("Nafn", "size"), Of_seint=("Late", "sum")).reset_index()
    diag = (
        df_races[["Round", "Race", "Brautarnafn", "Match_Track", "Start"]]
        .merge(counts, on="Match_Track", how="left")
        .fillna({"Spár": 0, "Of_seint": 0})
        .rename(columns={"Match_Track": "API circuitId", "Of_seint": "Of seint"})
    )
    diag["Start"] = diag["Start"].dt.strftime("%d.%m.%Y %H:%M UTC")
    sheet_tracks = (
        g.groupby(["Braut", "Match_Track"]).size().reset_index(name="Fjöldi")
        .rename(columns={"Braut": "Nafn í Sheets", "Match_Track": "Varpað á"})
    )
    sheet_tracks["Til í API"] = sheet_tracks["Varpað á"].isin(df_races["Match_Track"])

    return full, m, df_actual, df_races, players, entries_by_track, unknown_tracks, diag, sheet_tracks


def standings(df):
    t = df.groupby("Nafn").agg(
        Heildarstig=("Score", "sum"),
        Pows=("Pows", "sum"),
        Keppnir=("Status", lambda s: int((s == "Skilaði").sum())),
        Missti=("Status", lambda s: int((s == "Missti af").sum())),
    )
    return t.sort_values(["Heildarstig", "Pows"], ascending=[True, False])


# ==========================================
# 4. LOAD DATA (with fallback to the last good load)
# ==========================================
st.markdown("<h1 class='main-header'>🏎️ F1 SPÁDÓMSKEPPNIN 2026</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Aðmíraldeildin — Live Stigatafla & Tölfræði</p>",
            unsafe_allow_html=True)

with st.spinner("Sæki nýjustu stöðuna frá F1 API og Google Sheets..."):
    try:
        raw = load_guesses()
        races, fetched_at = fetch_races()
        st.session_state["last_good"] = (raw, races, fetched_at)
    except Exception as e:
        if "last_good" in st.session_state:
            raw, races, fetched_at = st.session_state["last_good"]
            st.warning(f"Gat ekki sótt ný gögn ({e}). Sýni síðustu gögn sem tókst að sækja.")
        else:
            st.error(f"Gat ekki sótt gögn: {e}")
            st.stop()

if not races:
    st.info("Engin keppni er búin ennþá — stigataflan birtist eftir fyrstu keppni.")
    st.stop()

(scores, guess_rows, df_actual, df_races, players, entries_by_track,
 unknown_tracks, diag, sheet_tracks) = compute(raw, races)

if scores.empty:
    st.info("Engar gildar spár fundust fyrir kláraðar keppnir.")
    st.stop()

# ==========================================
# 5. SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Stillingar")
if st.sidebar.button("🔄 Endurnýja Gögn"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"Gögn sótt: {fetched_at.strftime('%d.%m.%Y %H:%M')} UTC")

with st.sidebar.expander("📜 Reglur"):
    st.markdown(
        f"- Lægra er betra. Rétt sæti (**Pow**) gefur **{EXACT_HIT_SCORE}**, annars er skorið "
        f"munurinn á spáðu og raunverulegu sæti.\n"
        f"- Ökumaður utan úrslita gefur sæti {NOT_CLASSIFIED_POS}.\n"
        f"- Spá sem berst eftir að keppni hefst telst ekki með.\n"
        f"- Sá sem skilar ekki spá fær **versta skor keppninnar × {MISSED_MULTIPLIER}** (námundað)."
        + ("\n- Refsing gildir frá fyrstu spá hvers þátttakanda." if PENALTY_FROM_FIRST_ENTRY else "")
    )

if unknown_tracks:
    st.sidebar.warning("Óþekkt brautarnöfn í Sheets: " + ", ".join(unknown_tracks))

# ==========================================
# 6. NEXT RACE
# ==========================================
nxt = fetch_next_race()
if nxt:
    delta = pd.Timestamp(nxt["start"]) - pd.Timestamp.now(tz="UTC")
    when = f"{delta.days} d {delta.seconds // 3600} klst" if delta.total_seconds() > 0 else "í gangi / nýbúin"
    entered = entries_by_track.get(nxt["circuit"], [])
    waiting = [p for p in players if p not in entered]
    msg = f"🏁 **Næsta keppni:** {nxt['name']} — {when}. "
    msg += f"Búin að skila: {', '.join(entered) or '—'}. "
    msg += f"Vantar: {', '.join(waiting) or 'enginn 🎉'}."
    st.info(msg)

# ==========================================
# 7. LEADERBOARD
# ==========================================
cur = standings(scores)
submitted = scores[scores["Status"] == "Skilaði"]
cur = cur.join(submitted.groupby("Nafn")["Score"].agg(Meðaltal="mean", Besta="min", Versta="max"))
cur["Sæti"] = range(1, len(cur) + 1)

last_round = scores["Round"].max()
if scores["Round"].nunique() > 1:
    prev = standings(scores[scores["Round"] < last_round])
    prev_rank = pd.Series(range(1, len(prev) + 1), index=prev.index)
else:
    prev_rank = pd.Series(dtype=int)


def movement(name, rank):
    if name not in prev_rank.index:
        return "–"
    diff = int(prev_rank[name]) - rank
    return f"▲{diff}" if diff > 0 else f"▼{-diff}" if diff < 0 else "="


cur["Breyting"] = [movement(n, r) for n, r in zip(cur.index, cur["Sæti"])]

st.subheader("🏆 Núverandi Staða")
records = list(cur.reset_index().to_dict("records"))
for start in range(0, len(records), 6):
    chunk = records[start:start + 6]
    for col, row in zip(st.columns(len(chunk)), chunk):
        with col:
            st.metric(
                label=f"#{row['Sæti']} {row['Nafn']}",
                value=f"{int(row['Heildarstig'])} stig",
                delta=f"{int(row['Pows'])} Pows",
            )

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Heildarstaða", "🏎️ Einstakar Keppnir", "📈 Stigaþróun", "🔧 Greining"])

with tab1:
    st.subheader("Heildarstigataflan yfir tímabilið")
    table = cur.reset_index()[[
        "Sæti", "Breyting", "Nafn", "Heildarstig", "Pows", "Keppnir", "Missti",
        "Meðaltal", "Besta", "Versta"]]
    table["Heildarstig"] = table["Heildarstig"].astype(int)
    st.dataframe(
        table.style.highlight_min(subset=["Heildarstig"], color="#1e4620")
                   .format({"Meðaltal": "{:.1f}", "Besta": "{:.0f}", "Versta": "{:.0f}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption("Breyting = hreyfing á stöðu frá síðustu keppni. "
               "Missti = fjöldi keppna sem spá vantaði (refsing reiknuð). "
               "Meðaltal/Besta/Versta miðast aðeins við keppnir þar sem spá var skilað.")

    # --- podium of the most recent race ---
    st.markdown("---")
    last = df_races.sort_values("Round").iloc[-1]
    st.subheader(f"🏁 Verðlaunapallur — {last['Race']}")
    medals = ["🥇", "🥈", "🥉"]
    top3 = df_actual[
        (df_actual["Match_Track"] == last["Match_Track"]) & (df_actual["Actual_Pos"] <= 3)
    ].sort_values("Actual_Pos")
    for col, medal, row in zip(st.columns(3), medals, top3.itertuples()):
        with col:
            st.metric(label=f"{medal} P{row.Actual_Pos}", value=row.Actual_Driver)

    # --- top 3 of the drivers' championship ---
    st.markdown("---")
    st.subheader("🏆 Heimsmeistaramót ökumanna — topp 3")
    standings_top3 = fetch_driver_standings()
    if not standings_top3:
        st.caption("Staða ökumanna er ekki tiltæk í augnablikinu.")
    else:
        leader_points = standings_top3[0]["points"]
        for col, medal, s in zip(st.columns(3), medals, standings_top3):
            gap = leader_points - s["points"]
            with col:
                st.metric(
                    label=f"{medal} P{s['pos']} · {s['points']:g} stig",
                    value=s["driver"],
                    delta=f"-{gap:g} stig frá efsta" if gap > 0 else None,
                    delta_color="off",
                )

with tab2:
    st.subheader("Skoða ákveðna keppni")
    rounds = df_races.sort_values("Round")
    labels = {f"R{r.Round} · {r.Race}": r.Match_Track for r in rounds.itertuples()}
    selected = st.selectbox("Veldu keppni:", options=list(labels), index=len(labels) - 1)
    track = labels[selected]

    race_scores = scores[scores["Match_Track"] == track].sort_values(
        ["Score", "Pows"], ascending=[True, False])
    if race_scores.empty:
        st.warning("Enginn skilaði spá fyrir þessa keppni.")
    else:
        st.write(f"### {selected}")
        out = race_scores[["Nafn", "Score", "Pows", "Status"]].rename(
            columns={"Score": "Stig", "Status": "Staða"})
        out["Stig"] = out["Stig"].astype(int)
        st.dataframe(out, width="stretch", hide_index=True)

        g = guess_rows[guess_rows["Match_Track"] == track]
        vals = g.pivot(index="Predicted_Pos", columns="Nafn", values="Guess")
        diffs = g.pivot(index="Predicted_Pos", columns="Nafn", values="Difference")
        actual = df_actual[df_actual["Match_Track"] == track].set_index("Actual_Pos")["Actual_Driver"]
        vals.insert(0, "Úrslit", actual.reindex(vals.index))
        vals.index.name = "Sæti"

        def colour(_):
            css = pd.DataFrame("", index=vals.index, columns=vals.columns)
            for c in diffs.columns:
                css[c] = np.select(
                    [diffs[c] == 0, diffs[c] == 1],
                    ["background-color:#1e4620", "background-color:#5c4a00"],
                    default="")
            return css

        st.write("#### Spár vs. úrslit")
        st.dataframe(vals.style.apply(colour, axis=None), width="stretch")
        st.caption("🟩 rétt sæti · 🟨 einu sæti frá")

with tab3:
    st.subheader("Þróun yfir tímabilið")
    view = st.radio("Sýna:", ["Uppsöfnuð stig", "Sæti eftir keppni"], horizontal=True)

    chart = scores.sort_values(["Round", "Nafn"]).copy()
    chart["Uppsöfnuð stig"] = chart.groupby("Nafn")["Score"].cumsum()
    chart["Sæti eftir keppni"] = chart.groupby("Round")["Uppsöfnuð stig"].rank(method="min")
    order = rounds["Race"].tolist()

    fig = px.line(
        chart, x="Race", y=view, color="Nafn", markers=True,
        hover_data=["Status"],
        title=f"{view} (Lægra = Betra)",
        labels={"Race": "Keppnir"},
    )
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    if view == "Sæti eftir keppni":
        fig.update_yaxes(autorange="reversed", dtick=1)
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, width="stretch")

with tab4:
    st.subheader("Greining: pössun Sheets við F1 API")
    st.write("#### Keppnir í API")
    st.caption("Ef **Spár** er 0 fyrir keppni þá passar brautarnafnið í Sheets ekki við API. "
               "Ef **Of seint** er hátt var spám skilað eftir að keppni hófst (eða tímabeltið er rangt).")
    st.dataframe(diag, width="stretch", hide_index=True)
    st.write("#### Brautarnöfn í Sheets")
    st.caption("Raðir með **Til í API = False** tengjast engri kláraðri keppni "
               "(eðlilegt fyrir keppnir sem eru ekki búnar).")
    st.dataframe(sheet_tracks, width="stretch", hide_index=True)

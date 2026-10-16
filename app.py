import json
import streamlit as st
import gspread
import pandas as pd
import numpy as np
import requests
import unicodedata
import re
import plotly.express as px

# ==========================================
# 0. STYLED STREAMLIT PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="F1 Spádómskeppnin 2026",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS til að gefa síðunni F1 útlit
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FF1801;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #888;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def normalize_string(text):
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text

def normalize_driver_name(name):
    name = normalize_string(name)
    aliases = {
        "kimi antonelli": "andrea kimi antonelli",
        "alex albon": "alexander albon",
        "carlos sainz jr": "carlos sainz",
        "carlos sainz jr.": "carlos sainz",
        "yuki tsunoda": "tsunoda_hadjar_seat",
        "isack hadjar": "tsunoda_hadjar_seat"
    }
    if name in aliases:
        name = aliases[name]
    if "zhou" in name and "guanyu" in name:
        return "guanyu zhou"
    return name

def normalize_track(track_name):
    track = normalize_string(track_name)
    aliases = {
        "astralia": "albert_park",
        "kina": "shanghai",
        "japan": "suzuka",
        "barein": "sepang",
        "bahrain": "sepang",
        "bahrein": "sepang",
        "saudi arabia": "jeddah",
        "sadi arabia": "jeddah",
        "miami": "miami",
        "kanada": "villeneuve",
        "monako": "monaco",
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
        "madríd": "madring",
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
        "abu dhabi": "yas_marina"
    }
    for icelandic, api_id in aliases.items():
        if icelandic in track:
            return api_id
    return track

# ==========================================
# 2. CACHED DATA PIPELINE (FAST LOADING)
# ==========================================
@st.cache_data(ttl=600)
def load_and_calculate_data():
    SHEET_NAME = "Spádómskeppni (Responses)"

    # Sækir lykilorðin úr öruggri geymslu Streamlit
    creds_dict = json.loads(st.secrets["GCP_CREDENTIALS_JSON"])
    gc = gspread.service_account_from_dict(creds_dict)
    sheet = gc.open(SHEET_NAME).worksheet("Form Responses 1")
    data = sheet.get_all_records()

    df_guesses = pd.DataFrame(data)
    df_guesses = df_guesses[df_guesses['Nafn'] != '']

    if 'Timestamp' in df_guesses.columns:
        df_guesses = df_guesses.sort_values('Timestamp').drop_duplicates(subset=['Nafn', 'Braut'], keep='last')

    # Fetch actual results round-by-round
    actual_results = []
    for round_num in range(1, 25):
        url = f"https://api.jolpi.ca/ergast/f1/current/{round_num}/results.json"
        res = requests.get(url)
        if res.status_code != 200:
            break
        api_data = res.json()
        races = api_data['MRData']['RaceTable']['Races']
        if not races:
            break 
            
        race = races[0]
        circuit_id = race['Circuit']['circuitId']
        
        for driver in race['Results']:
            full_name = f"{driver['Driver']['givenName']} {driver['Driver']['familyName']}"
            actual_results.append({
                'Match_Track': circuit_id,
                'Match_Name': normalize_driver_name(full_name),
                'Actual_Pos': int(driver['position'])
            })

    df_actual = pd.DataFrame(actual_results)

    # Format guesses
    df_guesses_melted = df_guesses.melt(
        id_vars=['Nafn', 'Braut', 'Timestamp'], 
        value_vars=['1. sæti', '2. sæti', '3. sæti', '4. sæti', '5. sæti', '6. sæti', '7. sæti', '8. sæti'], 
        var_name='Predicted_Pos_Text',
        value_name='Driver_Guess_Original'
    )
    df_guesses_melted['Predicted_Pos'] = df_guesses_melted['Predicted_Pos_Text'].str.extract(r'(\d+)').astype(int)
    df_guesses_melted['Match_Name'] = df_guesses_melted['Driver_Guess_Original'].apply(normalize_driver_name)
    df_guesses_melted['Match_Track'] = df_guesses_melted['Braut'].apply(normalize_track)

    completed_tracks = df_actual['Match_Track'].unique()
    df_graded = df_guesses_melted[df_guesses_melted['Match_Track'].isin(completed_tracks)].copy()

    # Merge and calculate points
    df_final = pd.merge(df_graded, df_actual, on=['Match_Track', 'Match_Name'], how='left')
    df_final['Actual_Pos'] = df_final['Actual_Pos'].fillna(22)
    df_final['Difference'] = (df_final['Predicted_Pos'] - df_final['Actual_Pos']).abs()
    df_final['Score'] = np.where(df_final['Predicted_Pos'] == df_final['Actual_Pos'], -4, df_final['Difference'])
    df_final['Pows'] = np.where(df_final['Predicted_Pos'] == df_final['Actual_Pos'], 1, 0)

    return df_final, df_guesses['Braut'].unique()

# ==========================================
# 3. DASHBOARD DISPLAY
# ==========================================
st.markdown("<h1 class='main-header'>🏎️ F1 SPÁDÓMSKEPPNIN 2026</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Aðmíraldeildin — Live Stigatafla & Tölfræði</p>", unsafe_allow_html=True)

# Hlaða gögnum
with st.spinner("Sæki nýjustu stöðuna frá F1 API og Google Sheets..."):
    df_final, all_tracks = load_and_calculate_data()

# Hliðarstika (Sidebar) fyrir stjórntæki
st.sidebar.header("⚙️ Stillingar")
if st.sidebar.button("🔄 Endurnýja Gögn"):
    st.cache_data.clear()
    st.rerun()

# 🏆 HEILDARSTAÐA
leaderboard_total = df_final.groupby('Nafn').agg({
    'Score': 'sum',
    'Pows': 'sum'
}).reset_index().sort_values('Score')
leaderboard_total = leaderboard_total.rename(columns={'Score': 'Heildarstig', 'Pows': 'Heildar Pows'})

# Sýna yfirlitspjöld efst (Metric Cards)
st.subheader("🏆 Núverandi Staða")
cols = st.columns(len(leaderboard_total))
for idx, row in enumerate(leaderboard_total.itertuples()):
    with cols[idx]:
        st.metric(
            label=f"#{idx+1} {row.Nafn}", 
            value=f"{int(row.Heildarstig)} stig", 
            delta=f"{int(row._3)} Pows",
            delta_color="normal"
        )

st.markdown("---")

# FLIPPAR (TABS) FYRIR YFIRLIT
tab1, tab2, tab3 = st.tabs(["📊 Heildarstaða", "🏎️ Einstakir Keppnir", "📈 Stigaþróun"])

with tab1:
    st.subheader("Heildarstigatöfluna yfir tímabilið")
    st.dataframe(
        leaderboard_total.style.highlight_min(subset=['Heildarstig'], color='#1e4620'),
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.subheader("Skoða ákveðna braut")
    selected_track = st.selectbox("Veldu keppni:", options=all_tracks, index=len(all_tracks)-1)
    
    track_norm = normalize_track(selected_track)
    df_track = df_final[df_final['Match_Track'] == track_norm]
    
    if df_track.empty:
        st.warning("Niðurstöður fyrir þessa braut eru ekki komnar inn hjá F1 API ennþá.")
    else:
        race_lb = df_track.groupby('Nafn').agg({
            'Score': 'sum',
            'Pows': 'sum'
        }).reset_index().sort_values('Score')
        
        st.write(f"### Niðurstöður: {selected_track}")
        st.dataframe(race_lb, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Stigaþróun yfir tímabilið")
    
    # Reikna safnstik (cumulative scores) braut fyrir braut
    df_chart = df_final.groupby(['Match_Track', 'Nafn'])['Score'].sum().reset_index()
    df_chart['Kumulativ_Stig'] = df_chart.groupby('Nafn')['Score'].cumsum()
    
    fig = px.line(
        df_chart, 
        x='Match_Track', 
        y='Kumulativ_Stig', 
        color='Nafn',
        markers=True,
        title="Söfnun stiga eftir keppnum (Lægra = Betra)",
        labels={'Match_Track': 'Keppnir', 'Kumulativ_Stig': 'Heildarstig'}
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
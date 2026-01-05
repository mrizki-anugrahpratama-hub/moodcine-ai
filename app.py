import streamlit as st
import pandas as pd
import pickle
import time
import requests
import google.generativeai as genai
from deep_translator import GoogleTranslator
from sklearn.metrics.pairwise import linear_kernel

# ==========================================
# 1. KONFIGURASI & SETUP API
# ==========================================
st.set_page_config(
    page_title="MoodCine AI",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    h1, h2, h3 { color: #E50914; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { background-color: #E50914; color: white; border-radius: 20px; padding: 10px 24px; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #b20710; color: white; }
    .movie-card { background-color: #1f1f1f; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #333; transition: transform 0.2s; }
    .movie-card:hover { transform: scale(1.02); border-color: #E50914; }
    .stProgress > div > div > div > div { background-color: #E50914; }
    .highlight { color: #E50914; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Setup API Keys dari st.secrets (dengan Fallback aman)
TMDB_API_KEY = st.secrets.get("general", {}).get("tmdb_api_key", None)
GEMINI_API_KEY = st.secrets.get("general", {}).get("gemini_api_key", None)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# 2. LOAD RESOURCES (MODEL & DATA)
# ==========================================
@st.cache_resource
def load_resources():
    try:
        with open("C:/Users/Administrator/Desktop/FP Celerates/MoodCine AI/sentiment_model_tfidf.pkl", "rb") as f:
            model = pickle.load(f)
        with open("C:/Users/Administrator/Desktop/FP Celerates/MoodCine AI/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        movies = pd.read_csv("C:/Users/Administrator/Desktop/FP Celerates/MoodCine AI/processed_movies.csv")
        with open("C:/Users/Administrator/Desktop/FP Celerates/MoodCine AI/movie_vectorizer.pkl", "rb") as f:
            movie_tfidf = pickle.load(f)
        with open("C:/Users/Administrator/Desktop/FP Celerates/MoodCine AI/movie_tfidf_matrix.pkl", "rb") as f:
            movie_matrix = pickle.load(f)
        return model, vectorizer, movies, movie_tfidf, movie_matrix
    except FileNotFoundError:
        st.error("⚠️ File model tidak ditemukan. Pastikan Anda sudah menjalankan script training!")
        return None, None, None, None, None

sentiment_model, sentiment_vectorizer, df_movies, movie_tfidf, movie_matrix = load_resources()

# ==========================================
# 3. FUNGSI LOGIKA & API
# ==========================================

# --- A. Analisis Sentimen ---
def analyze_mood(text):
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
    except:
        translated = text # Fallback offline
    
    vec = sentiment_vectorizer.transform([translated])
    prediction = sentiment_model.predict(vec)[0]
    proba = sentiment_model.predict_proba(vec)[0]
    intensity = max(proba)
    
    return {
        "label": "Positif" if prediction == 1 else "Negatif",
        "score": intensity if prediction == 1 else -intensity,
        "translated_text": translated,
        "intensity": intensity
    }

# --- B. Rekomendasi Film (Cosine Similarity) ---
def get_recommendations(mood_text, user_genres, top_n=5):
    # Gabungkan mood (translated) + genre user untuk query pencarian
    query = f"{mood_text} {' '.join(user_genres)}"
    query_vec = movie_tfidf.transform([query])
    cosine_sim = linear_kernel(query_vec, movie_matrix).flatten()
    
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n]
    
    movie_indices = [i[0] for i in sim_scores]
    return df_movies.iloc[movie_indices]

# --- C. TMDB API (Ambil Poster) ---
def fetch_poster(movie_title):
    if not TMDB_API_KEY:
        return "https://via.placeholder.com/200x300?text=No+API+Key"
    
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        data = requests.get(url).json()
        if data['results']:
            poster_path = data['results'][0]['poster_path']
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return "https://via.placeholder.com/200x300?text=No+Poster"

# --- D. Gemini AI (Reasoning Psikologis) ---
def get_ai_reasoning(user_mood, movie_title, movie_overview):
    if not GEMINI_API_KEY:
        return "Fitur AI Reasoning tidak aktif (API Key belum diset)."
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = (
            f"User sedang merasa: '{user_mood}'. "
            f"Saya merekomendasikan film '{movie_title}' yang bercerita tentang: '{movie_overview}'. "
            "Berikan alasan singkat (maksimal 2 kalimat) dalam Bahasa Indonesia yang santai dan empati, "
            "mengapa film ini cocok untuk menemani perasaan user tersebut. Jangan kaku."
        )
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Film ini memiliki tone cerita yang relevan dengan mood kamu."

# ==========================================
# 4. NAVIGASI HALAMAN
# ==========================================
if 'page' not in st.session_state: st.session_state.page = 0
if 'user_data' not in st.session_state: st.session_state.user_data = {"name": "", "mood": "", "genres": []}

def next_page(): st.session_state.page += 1
def restart(): 
    st.session_state.page = 0
    st.session_state.user_data = {"name": "", "mood": "", "genres": []}

# ==========================================
# 5. UI FLOW
# ==========================================

# PAGE 0: WELCOME
if st.session_state.page == 0:
    st.title("🎬 MoodCine AI")
    st.markdown("### Temukan Film Sesuai Isi Hatimu")
    st.write("Aplikasi ini menggunakan **AI Sentimen Analisis** untuk memahami emosimu dan mencarikan film yang paling pas.")
    st.markdown("---")
    if st.button("Mulai Sekarang 🚀", use_container_width=True): next_page()

# PAGE 1: NAMA
elif st.session_state.page == 1:
    st.title("👋 Kenalan Dulu")
    name = st.text_input("Siapa nama panggilanmu?", value=st.session_state.user_data["name"])
    if st.button("Lanjut") and name:
        st.session_state.user_data["name"] = name
        next_page()

# PAGE 2: MOOD INPUT
elif st.session_state.page == 2:
    st.title(f"Apa kabar, {st.session_state.user_data['name']}?")
    mood_text = st.text_area("Ceritakan harimu (Bahasa Indonesia):", placeholder="Misal: Lagi capek banget kerja, butuh hiburan yang bikin ketawa.")
    
    st.write("Atau pilih mood cepat:")
    chips = ["Senang & Semangat 😆", "Lelah & Butuh Istirahat 😴", "Sedih & Galau 😢", "Marah & Kesal 😠", "Bosan & Gabut 😐"]
    try:
        selected_chip = st.pills("Tags:", chips)
    except:
        selected_chip = st.radio("Tags:", chips) # Fallback

    if st.button("Lanjut"):
        final_mood = mood_text if mood_text else selected_chip
        if final_mood:
            st.session_state.user_data["mood"] = final_mood
            next_page()
        else:
            st.warning("Isi dulu ya perasaannya!")

# PAGE 3: GENRE
elif st.session_state.page == 3:
    st.title("Genre Favorit?")
    genres = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi"]
    selected = st.multiselect("Pilih maksimal 3 genre (Opsional):", genres, max_selections=3)
    
    if st.button("Analisa Mood Saya 🔮", use_container_width=True):
        st.session_state.user_data["genres"] = selected
        next_page()

# PAGE 4: LOADING (PROCESS)
elif st.session_state.page == 4:
    st.title("🔄 Sedang Memproses...")
    bar = st.progress(0)
    status = st.empty()
    
    # Step 1: Sentimen
    status.write("🧠 Menerjemahkan & Menganalisis emosi...")
    mood_text = st.session_state.user_data["mood"]
    analysis = analyze_mood(mood_text)
    bar.progress(30)
    
    # Step 2: Rekomendasi
    status.write("🔍 Mencari film di database...")
    recs = get_recommendations(analysis['translated_text'], st.session_state.user_data["genres"])
    bar.progress(60)
    
    # Step 3: Fetch Data Eksternal (Poster)
    # Kita fetch di halaman hasil saja biar loading page ini cepat, atau fetch disini sebagian
    status.write("🎨 Menyiapkan poster & visual...")
    time.sleep(1)
    bar.progress(100)
    
    st.session_state.results = {"analysis": analysis, "recommendations": recs}
    next_page()
    st.rerun()

# PAGE 5: HASIL
elif st.session_state.page == 5:
    res = st.session_state.results
    user = st.session_state.user_data
    
    st.title("✨ Rekomendasi Untukmu")
    
    # Header Info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mood Terdeteksi", res['analysis']['label'])
    with col2:
        st.metric("Intensitas", f"{res['analysis']['intensity']*100:.0f}%")
        
    st.info(f"**Analisa:** Kamu sepertinya merasa *{user['mood']}*. Sistem mencari film yang cocok dengan vibe tersebut.")
    
    # Tampilkan Film
    for index, row in res['recommendations'].iterrows():
        with st.container():
            st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
            c_img, c_info = st.columns([1, 2])
            
            with c_img:
                poster_url = fetch_poster(row['title'])
                st.image(poster_url, use_container_width=True)
                
            with c_info:
                st.subheader(f"{row['title']} ({str(row['release_date'])[:4]})")
                st.caption(f"⭐ {row['vote_average']}/10 | {row['genres']}")
                st.write(f"_{row['overview'][:150]}..._")
                
                # Expandable AI Reasoning
                with st.expander("🤖 Kata AI tentang film ini..."):
                    # Panggil Gemini on-the-fly (cache result jika perlu biar hemat kuota)
                    reasoning = get_ai_reasoning(user['mood'], row['title'], row['overview'])
                    st.write(reasoning)
                    
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Coba Lagi 🔄", use_container_width=True):
        restart()
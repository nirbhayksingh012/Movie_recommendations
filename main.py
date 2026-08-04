import os
import pickle
import asyncio
from typing import Optional, List, Dict, Any, Tuple

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import httpx
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Query
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
OMDB_BASE = "http://www.omdbapi.com/"
if not OMDB_API_KEY:
    raise RuntimeError("OMDB_API_KEY missing. Put it in .env as OMDB_API_KEY=xxxx")

app = FastAPI(title="Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# PICKLE GLOBALS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# MODELS
# =========================
class OMDBMovieCard(BaseModel):
    imdb_id: str
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class OMDBMovieDetails(BaseModel):
    imdb_id: str
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    genres: List[str] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    omdb: Optional[OMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: OMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[OMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(url: Optional[str]) -> Optional[str]:
    """OMDB returns full poster URLs directly, or 'N/A' if not available."""
    if not url or url == "N/A":
        return None
    return url


def get_local_movie_details(imdb_id: str) -> Optional[OMDBMovieDetails]:
    global df
    if df is None:
        return None
    norm_id = str(imdb_id).strip().lower()
    matches = df[df['imdb_id'].fillna('').astype(str).str.strip().str.lower() == norm_id]
    if matches.empty:
        return None
    row = matches.iloc[0]
    genres_str = row.get('genres', '')
    genres = [g.strip() for g in genres_str.split() if g.strip()]
    poster_path = row.get('poster_path')
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path and str(poster_path) != "nan" else None
    
    return OMDBMovieDetails(
        imdb_id=str(row.get('imdb_id', imdb_id)),
        title=str(row.get('title', 'Untitled')),
        overview=str(row.get('overview', '')),
        release_date=str(row.get('release_date', '')),
        poster_url=poster_url,
        genres=genres
    )


def get_local_movie_card_by_title(title: str) -> Optional[OMDBMovieCard]:
    global df
    if df is None:
        return None
    norm_title = _norm_title(title)
    matches = df[df['title'].fillna('').astype(str).str.strip().str.lower() == norm_title]
    if matches.empty:
        return None
    row = matches.iloc[0]
    poster_path = row.get('poster_path')
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path and str(poster_path) != "nan" else None
    return OMDBMovieCard(
        imdb_id=str(row.get('imdb_id', '')),
        title=str(row.get('title', title)),
        poster_url=poster_url,
        release_date=str(row.get('release_date', '')),
        vote_average=float(row.get('vote_average', 0.0))
    )


# In-memory poster cache to avoid redundant network requests
POSTER_CACHE: Dict[str, str] = {}


async def verify_and_get_poster(title: str, imdb_id: str, local_path: Optional[str]) -> Optional[str]:
    global POSTER_CACHE
    
    # 0. Check cache first
    cache_key = str(imdb_id or title).strip().lower()
    if cache_key in POSTER_CACHE:
        return POSTER_CACHE[cache_key]

    # 1. Try local TMDB path first
    if local_path and str(local_path) != "nan" and str(local_path).strip():
        url = f"https://image.tmdb.org/t/p/w500{local_path}"
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.head(url)
                if r.status_code == 200:
                    POSTER_CACHE[cache_key] = url
                    return url
        except Exception:
            pass

    # 2. Keyless Free Movie DB API Fallback (using title)
    if title and str(title).strip() and str(title) != "nan":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"https://imdb.iamidiotareyoutoo.com/search?q={title}")
                if r.status_code == 200:
                    data = r.json()
                    if data.get("ok") and data.get("description"):
                        results = data["description"]
                        # Match exact IMDb ID
                        for item in results:
                            item_imdb = item.get("#IMDB_ID")
                            if imdb_id and item_imdb and item_imdb.strip().lower() == imdb_id.strip().lower():
                                poster = item.get("#IMG_POSTER")
                                if poster and poster != "N/A":
                                    POSTER_CACHE[cache_key] = poster
                                    return poster
                        # Fallback to the first matching title result
                        poster = results[0].get("#IMG_POSTER")
                        if poster and poster != "N/A":
                            POSTER_CACHE[cache_key] = poster
                            return poster
        except Exception:
            pass

    # 3. Fallback to OMDB if configured
    if imdb_id and str(imdb_id).strip() and str(imdb_id) != "nan" and OMDB_API_KEY and OMDB_API_KEY != "f0891f70":
        try:
            data = await omdb_get({"i": imdb_id})
            poster = data.get("Poster")
            if poster and poster != "N/A":
                POSTER_CACHE[cache_key] = poster
                return poster
        except Exception:
            pass

    return None


async def local_genre_recommend(imdb_id: str, limit: int = 18) -> List[OMDBMovieCard]:
    global df
    if df is None:
        return []
    
    norm_id = str(imdb_id).strip().lower()
    matches = df[df['imdb_id'].fillna('').astype(str).str.strip().str.lower() == norm_id]
    if matches.empty:
        return []
    
    target_movie = matches.iloc[0]
    target_genres_str = target_movie.get('genres', '')
    target_genres = set(target_genres_str.split())
    
    if not target_genres:
        return []
    
    overlap_scores = []
    for idx, row in df.iterrows():
        if str(row.get('imdb_id')) == target_movie.get('imdb_id'):
            continue
        row_genres = set(str(row.get('genres', '')).split())
        overlap = len(target_genres.intersection(row_genres))
        if overlap > 0:
            popularity = float(row.get('popularity', 0.0))
            overlap_scores.append((idx, overlap, popularity))
            
    if not overlap_scores:
        return []
    
    overlap_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    async def build_card(idx):
        row = df.iloc[idx]
        row_imdb = str(row.get('imdb_id', ''))
        poster_path = row.get('poster_path')
        poster_url = await verify_and_get_poster(str(row.get('title', '')), row_imdb, poster_path)
        return OMDBMovieCard(
            imdb_id=row_imdb,
            title=str(row.get('title', '')),
            poster_url=poster_url,
            release_date=str(row.get('release_date', '')),
            vote_average=float(row.get('vote_average', 0.0))
        )
        
    tasks = [build_card(idx) for idx, overlap, pop in overlap_scores[:limit]]
    return await asyncio.gather(*tasks)


async def omdb_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe OMDB GET:
    - Network errors -> 502
    - OMDB API errors -> 502 with detail
    """
    q = dict(params)
    q["apikey"] = OMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(OMDB_BASE, params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"OMDB error {r.status_code}: {r.text}"
        )

    data = r.json()
    if data.get("Response") == "False":
        raise HTTPException(
            status_code=404, detail=f"OMDB: {data.get('Error', 'Unknown error')}"
        )
    return data


async def omdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[OMDBMovieCard]:
    out: List[OMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            OMDBMovieCard(
                imdb_id=m.get("imdbID") or "",
                title=m.get("Title") or "",
                poster_url=make_img_url(m.get("Poster")),
                release_date=m.get("Year"),
                vote_average=None,
            )
        )
    return out


async def omdb_movie_details(imdb_id: str) -> OMDBMovieDetails:
    """Fetch movie details from OMDB by IMDb ID."""
    data = await omdb_get({"i": imdb_id, "type": "movie", "plot": "full"})
    # Parse genres (OMDB returns comma-separated string)
    genres_str = data.get("Genre", "")
    genres = [g.strip() for g in genres_str.split(",") if g.strip()]
    return OMDBMovieDetails(
        imdb_id=data.get("imdbID") or imdb_id,
        title=data.get("Title") or "",
        overview=data.get("Plot"),
        release_date=data.get("Year"),
        poster_url=make_img_url(data.get("Poster")),
        genres=genres,
    )


async def omdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Search OMDB for movies by title.
    OMDB returns: {Search: [{Title, Year, imdbID, Type, Poster}, ...], Response, totalResults}
    """
    return await omdb_get(
        {
            "s": query,
            "type": "movie",
            "page": page,
        },
    )


async def omdb_search_first(query: str) -> Optional[dict]:
    """Return first search result from OMDB."""
    data = await omdb_search_movies(query=query, page=1)
    results = data.get("Search", [])
    return results[0] if results else None


# =========================
# TF-IDF Helpers
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    # pandas Series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        # last resort: if it's a list-like etc.
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns list of (title, score) from local df using cosine similarity on TF-IDF matrix.
    Safe against missing columns/rows.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    # query vector
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    # sort descending
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_omdb_card_by_title(title: str) -> Optional[OMDBMovieCard]:
    """
    Uses local lookup or OMDB search by title to fetch poster for a local title.
    """
    global df
    if df is None:
        return None
    
    norm_title = _norm_title(title)
    matches = df[df['title'].fillna('').astype(str).str.strip().str.lower() == norm_title]
    if matches.empty:
        try:
            m = await omdb_search_first(title)
            if not m:
                return None
            return OMDBMovieCard(
                imdb_id=m.get("imdbID") or "",
                title=m.get("Title") or title,
                poster_url=make_img_url(m.get("Poster")),
                release_date=m.get("Year"),
                vote_average=None,
            )
        except Exception:
            return None
            
    row = matches.iloc[0]
    imdb_id = str(row.get('imdb_id', ''))
    poster_path = row.get('poster_path')
    poster_url = await verify_and_get_poster(str(row.get('title', '')), imdb_id, poster_path)
    
    return OMDBMovieCard(
        imdb_id=imdb_id,
        title=str(row.get('title', title)),
        poster_url=poster_url,
        release_date=str(row.get('release_date', '')),
        vote_average=float(row.get('vote_average', 0.0))
    )


# =========================
# STARTUP: LOAD PICKLES
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    # Load df
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    # Load indices
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    # Load TF-IDF matrix (usually scipy sparse)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Load tfidf vectorizer (optional, not used directly here)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    # Build normalized map
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    # sanity
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- HOME FEED (Local Dataset discovery) ----------
@app.get("/home", response_model=List[OMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Home feed for Streamlit (posters), served locally.
    Supports popular, trending, top_rated, now_playing, upcoming.
    """
    global df
    if df is None:
        return []
    
    subset = df.copy()
    
    # Sort or filter based on category
    if category in ("popular", "trending"):
        subset = subset.sort_values(by="popularity", ascending=False)
    elif category == "top_rated":
        # require minimum vote count to avoid skewing by minor films
        min_votes = 100
        high_votes_subset = subset[subset['vote_count'] >= min_votes]
        if high_votes_subset.empty:
            high_votes_subset = subset
        subset = high_votes_subset.sort_values(by="vote_average", ascending=False)
    elif category in ("now_playing", "upcoming"):
        # filter for valid year formats and sort by date/popularity
        subset = subset[subset['release_date'].fillna('').astype(str).str.contains(r'^\d{4}')]
        subset = subset.sort_values(by=["release_date", "popularity"], ascending=[False, False])
    else:
        subset = subset.sort_values(by="popularity", ascending=False)
    
    async def build_card(row):
        row_imdb = str(row.get('imdb_id', ''))
        poster_path = row.get('poster_path')
        poster_url = await verify_and_get_poster(str(row.get('title', '')), row_imdb, poster_path)
        return OMDBMovieCard(
            imdb_id=row_imdb,
            title=str(row.get('title', '')),
            poster_url=poster_url,
            release_date=str(row.get('release_date', '')),
            vote_average=float(row.get('vote_average', 0.0))
        )
        
    tasks = [build_card(row) for _, row in subset.head(limit).iterrows()]
    return await asyncio.gather(*tasks)


# ---------- OMDB KEYWORD SEARCH WITH LOCAL FALLBACK ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Returns search results from OMDB, falling back to local titles if OMDB fails.
    """
    try:
        return await omdb_search_movies(query=query, page=page)
    except Exception as e:
        # local search fallback
        global df
        if df is None:
            raise e
        query_l = query.strip().lower()
        matches = df[df['title'].fillna('').astype(str).str.lower().str.contains(query_l)]
        matches = matches.sort_values(by="popularity", ascending=False)
        
        limit = 10
        start = (page - 1) * limit
        paginated = matches.iloc[start:start+limit]
        
        results = []
        for _, row in paginated.iterrows():
            results.append({
                "Title": str(row.get('title', '')),
                "Year": str(row.get('release_date', ''))[:4],
                "imdbID": str(row.get('imdb_id', '')),
                "Type": "movie",
                "Poster": f"https://image.tmdb.org/t/p/w500{row.get('poster_path')}" if row.get('poster_path') and str(row.get('poster_path')) != "nan" else "N/A"
            })
        return {
            "Search": results,
            "totalResults": str(len(matches)),
            "Response": "True" if results else "False"
        }


# ---------- MOVIE DETAILS (WITH LOCAL FALLBACK) ----------
@app.get("/movie/id/{imdb_id}", response_model=OMDBMovieDetails)
async def movie_details_route(imdb_id: str):
    try:
        return await omdb_movie_details(imdb_id)
    except Exception as e:
        # local details lookup fallback
        local_details = get_local_movie_details(imdb_id)
        if local_details:
            return local_details
        raise e


# ---------- LOCAL GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[OMDBMovieCard])
async def recommend_genre(
    imdb_id: str = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """
    Return local genre-based movie recommendations.
    """
    return await local_genre_recommend(imdb_id, limit)


# ---------- TF-IDF ONLY (debug/useful) ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF recs + Genre recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    This endpoint is for when you have a selected movie and want:
      - movie details
      - TF-IDF recommendations (local) + posters
      - Genre recommendations (local) + posters
    """
    # Try fetching details from OMDB first, fallback to local lookup by title query
    imdb_id = None
    details = None
    
    try:
        best = await omdb_search_first(query)
        if best:
            imdb_id = best.get("imdbID")
            if imdb_id:
                details = await omdb_movie_details(imdb_id)
    except Exception:
        pass
        
    if not details:
        # Local fallback by title search
        global df
        if df is not None:
            norm_q = _norm_title(query)
            matches = df[df['title'].fillna('').astype(str).str.strip().str.lower() == norm_q]
            if not matches.empty:
                imdb_id = str(matches.iloc[0].get('imdb_id', ''))
                if imdb_id:
                    details = get_local_movie_details(imdb_id)

    if not details:
        raise HTTPException(
            status_code=404, detail=f"No movie found locally or via OMDB for query: {query}"
        )

    imdb_id = details.imdb_id

    # 1) TF-IDF recommendations (never crash endpoint)
    tfidf_items: List[TFIDFRecItem] = []
    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    tasks = [attach_omdb_card_by_title(title) for title, score in recs]
    cards = await asyncio.gather(*tasks)
    
    for (title, score), card in zip(recs, cards):
        tfidf_items.append(TFIDFRecItem(title=title, score=score, omdb=card))

    # 2) Genre recommendations (Served locally)
    genre_recs = await local_genre_recommend(imdb_id, limit=genre_limit)

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )

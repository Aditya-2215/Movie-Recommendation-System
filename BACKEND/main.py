from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from recommender import MovieRecommender
from tmdb_service import TMDBService
from config import settings

recommender = MovieRecommender(settings.MODELS_DIR)
tmdb = TMDBService(settings.TMDB_API_KEY, settings.TMDB_BEARER_TOKEN)


@asynccontextmanager
async def lifespan(app: FastAPI):
    recommender.load_models()
    yield
    await tmdb.close()


app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommendation engine using TF-IDF, cosine similarity and TMDB metadata.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Movie Recommendation API is running",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_loaded": recommender.is_loaded,
        "movie_count": recommender.movie_count,
        "tmdb_configured": tmdb.configured,
    }


@app.get("/api/search")
def search_movies(
    q: str = Query(..., min_length=1, description="Movie title to search"),
    limit: int = Query(10, ge=1, le=20),
    use_tmdb: bool = Query(True),
):
    local_results = recommender.search_titles(q, limit=limit)

    if not use_tmdb or not tmdb.configured:
        return {"source": "local", "results": local_results}

    try:
        tmdb_results = tmdb.search_movies(q, limit=limit)
        return {"source": "tmdb", "results": tmdb_results, "local_results": local_results}
    except Exception:
        return {"source": "local", "results": local_results}


@app.get("/api/recommend")
async def recommend(
    title: str = Query(..., min_length=1),
    n: int = Query(10, ge=1, le=30),
    enrich_tmdb: bool = Query(True),
):
    try:
        results = recommender.recommend(title, n=n)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if enrich_tmdb and tmdb.configured:
        results = await tmdb.enrich_recommendations(results)

    source_movie = recommender.get_movie(title)

    return {
        "query": source_movie,
        "count": len(results),
        "method": "TF-IDF + cosine similarity",
        "recommendations": results,
    }


@app.get("/api/movie/{tmdb_id}")
async def movie_details(tmdb_id: int):
    if not tmdb.configured:
        raise HTTPException(status_code=503, detail="TMDB is not configured")

    try:
        return await tmdb.get_movie_details(tmdb_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TMDB request failed: {exc}") from exc

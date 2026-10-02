import asyncio
from typing import Any

import httpx


class TMDBService:
    BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
    BACKDROP_BASE_URL = "https://image.tmdb.org/t/p/w1280"

    def __init__(self, api_key: str = "", bearer_token: str = ""):
        self.api_key = api_key.strip()
        self.bearer_token = bearer_token.strip()
        self.client = httpx.AsyncClient(timeout=10.0)
        self.sync_client = httpx.Client(timeout=10.0)

    @property
    def configured(self) -> bool:
        return bool(self.api_key or self.bearer_token)

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    def _params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        if self.api_key and not self.bearer_token:
            params["api_key"] = self.api_key
        return params

    async def search_movies_async(self, query: str, limit: int = 10) -> list[dict]:
        if not self.configured:
            return []

        response = await self.client.get(
            f"{self.BASE_URL}/search/movie",
            params=self._params({
                "query": query,
                "language": "en-US",
                "include_adult": "false",
                "page": 1,
            }),
            headers=self._headers(),
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for movie in data.get("results", [])[:limit]:
            results.append(self._format_movie(movie))
        return results

    def search_movies(self, query: str, limit: int = 10) -> list[dict]:
        if not self.configured:
            return []

        response = self.sync_client.get(
            f"{self.BASE_URL}/search/movie",
            params=self._params({
                "query": query,
                "language": "en-US",
                "include_adult": "false",
                "page": 1,
            }),
            headers=self._headers(),
        )
        response.raise_for_status()
        data = response.json()
        return [self._format_movie(movie) for movie in data.get("results", [])[:limit]]

    async def get_movie_details(self, tmdb_id: int) -> dict:
        if not self.configured:
            raise RuntimeError("TMDB is not configured.")

        response = await self.client.get(
            f"{self.BASE_URL}/movie/{tmdb_id}",
            params=self._params({
                "language": "en-US",
                "append_to_response": "credits,videos",
            }),
            headers=self._headers(),
        )
        response.raise_for_status()
        return self._format_details(response.json())

    async def enrich_recommendations(self, recommendations: list[dict]) -> list[dict]:
        if not self.configured or not recommendations:
            return recommendations

        # If the local dataset contains TMDB IDs, use them directly.
        tasks = []
        valid_positions = []
        for position, movie in enumerate(recommendations):
            tmdb_id = self._parse_id(movie.get("id"))
            if tmdb_id is not None:
                tasks.append(self.get_movie_details(tmdb_id))
                valid_positions.append(position)
            else:
                tasks.append(self.search_movies_async(movie.get("title", ""), limit=1))
                valid_positions.append(position)

        # Keep TMDB requests concurrent but bounded.
        semaphore = asyncio.Semaphore(5)

        async def limited(task):
            async with semaphore:
                try:
                    return await task
                except Exception:
                    return None

        responses = await asyncio.gather(*(limited(task) for task in tasks))

        for position, metadata in zip(valid_positions, responses):
            if not metadata:
                continue

            if isinstance(metadata, list):
                if metadata:
                    metadata = metadata[0]
                else:
                    continue

            recommendations[position].update({
                key: value
                for key, value in metadata.items()
                if value is not None
            })

        return recommendations

    @staticmethod
    def _parse_id(value: Any) -> int | None:
        try:
            number = int(float(value))
            return number if number > 0 else None
        except (TypeError, ValueError):
            return None

    def _format_movie(self, movie: dict) -> dict:
        poster = movie.get("poster_path")
        backdrop = movie.get("backdrop_path")
        release_date = movie.get("release_date") or ""

        return {
            "tmdb_id": movie.get("id"),
            "title": movie.get("title") or movie.get("original_title") or "Unknown",
            "original_title": movie.get("original_title"),
            "overview": movie.get("overview") or "",
            "release_date": release_date,
            "year": release_date[:4] if release_date else None,
            "rating": movie.get("vote_average", 0),
            "vote_count": movie.get("vote_count", 0),
            "popularity": movie.get("popularity", 0),
            "poster_path": poster,
            "poster_url": f"{self.IMAGE_BASE_URL}{poster}" if poster else None,
            "backdrop_path": backdrop,
            "backdrop_url": f"{self.BACKDROP_BASE_URL}{backdrop}" if backdrop else None,
        }

    def _format_details(self, movie: dict) -> dict:
        result = self._format_movie(movie)

        result.update({
            "tagline": movie.get("tagline") or "",
            "runtime": movie.get("runtime"),
            "status": movie.get("status"),
            "genres": [genre.get("name") for genre in movie.get("genres", [])],
            "homepage": movie.get("homepage"),
        })

        credits = movie.get("credits", {})
        cast = credits.get("cast", [])[:10]
        crew = credits.get("crew", [])

        directors = [
            person.get("name")
            for person in crew
            if person.get("job") == "Director"
        ]

        result["cast"] = [
            {
                "name": person.get("name"),
                "character": person.get("character"),
                "profile_url": (
                    f"{self.IMAGE_BASE_URL}{person.get('profile_path')}"
                    if person.get("profile_path") else None
                ),
            }
            for person in cast
        ]
        result["directors"] = directors

        videos = movie.get("videos", {}).get("results", [])
        trailers = [
            video for video in videos
            if video.get("site") == "YouTube" and video.get("type") == "Trailer"
        ]
        if trailers:
            result["trailer_url"] = f"https://www.youtube.com/watch?v={trailers[0].get('key')}"
        else:
            result["trailer_url"] = None

        return result

    async def close(self):
        await self.client.aclose()
        self.sync_client.close()

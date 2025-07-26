# Code from: https://github.com/sjanaX01/weather-mcp-server/
from mcp.server.fastmcp import FastMCP
import httpx
import os
from dotenv import load_dotenv
from fastapi import HTTPException
from datetime import datetime
import logging

# Load environment variables from .env file
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("WeatherMCP")

# Create an MCP server named "WeatherMCP"
mcp = FastMCP(
    name="WeatherMCP",
    prompt="This is a Weather server. You can get current weather, forecast, air quality, and astronomy information by calling the available tools.",
)


# Helper: call WeatherAPI asynchronously
def validate_date(dt_str: str) -> None:
    """
    Ensure date string is in YYYY-MM-DD format.
    Raises HTTPException if invalid.
    """
    try:
        datetime.strptime(dt_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid date: {dt_str}. Use YYYY-MM-DD."
        )


async def fetch(endpoint: str, params: dict) -> dict:
    """
    Perform async GET to WeatherAPI and return JSON.
    Raises HTTPException on errors.
    Enhanced: logs requests, handles non-JSON errors gracefully.
    """
    if not WEATHER_API_KEY:
        logger.error("Weather API key not set.")
        raise HTTPException(status_code=500, detail="Weather API key not set.")

    params["key"] = WEATHER_API_KEY
    url = f"https://api.weatherapi.com/v1/{endpoint}"
    logger.info(f"Requesting {url} with params {params}")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, params=params)
            try:
                data = resp.json()
            except Exception:
                data = None
            if resp.status_code != 200:
                detail = (data or {}).get("error", {}).get("message", resp.text)
                logger.error(f"WeatherAPI error {resp.status_code}: {detail}")
                raise HTTPException(status_code=resp.status_code, detail=detail)
            logger.info(f"WeatherAPI success: {url}")
            return data
        except httpx.RequestError as e:
            logger.error(f"HTTPX request error: {e}")
            raise HTTPException(status_code=500, detail=f"Request error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# MCP Tools


@mcp.tool()
async def weather_current(q: str, aqi: str = "no") -> dict:
    """
    Get current weather for a location.
    Args:
        q (str): Location query (city name, lat/lon, postal code, etc).
        aqi (str): Include air quality data ('yes' or 'no').
    Returns:
        dict: WeatherAPI current weather JSON.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Location (q) is required.")
    return await fetch("current.json", {"q": q, "aqi": aqi})


@mcp.tool()
async def weather_forecast(
    q: str, days: int = 1, aqi: str = "no", alerts: str = "no"
) -> dict:
    """
    Get weather forecast (1–14 days) for a location.
    Args:
        q (str): Location query (city name, lat/lon, postal code, etc).
        days (int): Number of days (1–14).
        aqi (str): Include air quality ('yes' or 'no').
        alerts (str): Include weather alerts ('yes' or 'no').
    Returns:
        dict: WeatherAPI forecast JSON.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Location (q) is required.")
    if days < 1 or days > 14:
        raise HTTPException(status_code=400, detail="'days' must be between 1 and 14.")
    return await fetch(
        "forecast.json", {"q": q, "days": days, "aqi": aqi, "alerts": alerts}
    )


@mcp.tool()
async def weather_history(q: str, dt: str) -> dict:
    """
    Get historical weather for a location on a given date (YYYY-MM-DD).
    Args:
        q (str): Location query.
        dt (str): Date in YYYY-MM-DD format.
    Returns:
        dict: WeatherAPI history JSON.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Location (q) is required.")
    validate_date(dt)
    return await fetch("history.json", {"q": q, "dt": dt})


@mcp.tool()
async def weather_alerts(q: str) -> dict:
    """
    Get weather alerts for a location.
    Args:
        q (str): Location query.
    Returns:
        dict: WeatherAPI alerts JSON.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Location (q) is required.")
    # Alerts come from forecast with alerts=yes
    return await fetch("forecast.json", {"q": q, "days": 1, "alerts": "yes"})


@mcp.tool()
async def weather_airquality(q: str) -> dict:
    """
    Get air quality for a location.
    Args:
        q (str): Location query.
    Returns:
        dict: WeatherAPI air quality JSON.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Location (q) is required.")
    return await fetch("current.json", {"q": q, "aqi": "yes"})


@mcp.tool()
async def weather_astronomy(q: str, dt: str) -> dict:
    """
    Get astronomy data (sunrise, sunset, moon) for a date (YYYY-MM-DD).
    Args:
        q (str): Location query.
        dt (str): Date in YYYY-MM-DD format.
    Returns:
        dict: WeatherAPI astronomy JSON.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Location (q) is required.")
    validate_date(dt)
    return await fetch("astronomy.json", {"q": q, "dt": dt})


@mcp.tool()
async def weather_search(q: str) -> dict:
    """
    Search for locations matching query.
    Args:
        q (str): Location query.
    Returns:
        dict: WeatherAPI search JSON.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Location (q) is required.")
    return await fetch("search.json", {"q": q})


@mcp.tool()
async def weather_timezone(q: str) -> dict:
    """
    Get timezone info for a location.
    Args:
        q (str): Location query.
    Returns:
        dict: WeatherAPI timezone JSON.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Location (q) is required.")
    return await fetch("timezone.json", {"q": q})


# Run the MCP server
if __name__ == "__main__":
    # This starts a Server-Sent Events (SSE) endpoint on port 8000
    mcp.run()

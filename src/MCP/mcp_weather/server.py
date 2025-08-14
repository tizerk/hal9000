# Code mostly from: https://github.com/sjanaX01/weather-mcp-server/
from mcp.server.fastmcp import FastMCP
import httpx
import os
from dotenv import load_dotenv
from fastapi import HTTPException
from datetime import datetime
import logging
from rich.logging import RichHandler

load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

mcp = FastMCP(
    name="WeatherMCP",
    prompt="This is a Weather server. You can get current weather, alerts, and air quality information by calling the available tools.",
)


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
async def weather_forecast(q: str, aqi: str = "no") -> dict:
    """
    Get tomorrow's weather for a location.
    Args:
        q (str): Location query (city name, lat/lon, postal code, etc).
        aqi (str): Include air quality data ('yes' or 'no').
    Returns:
        dict: WeatherAPI weather forecast with the sky conditions and high/low temperatures of the day.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Location (q) is required.")
    response = await fetch("forecast.json", {"q": q, "aqi": aqi})
    if "current" in response:
        response.pop("current")
    if "hour" in response["forecast"]["forecastday"][0]:
        response["forecast"]["forecastday"][0].pop("hour")
    return response


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
    response = await fetch("forecast.json", {"q": q, "days": 1, "alerts": "yes"})
    return response["alerts"]


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


# Run the MCP server
if __name__ == "__main__":
    mcp.run()

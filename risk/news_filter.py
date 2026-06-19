import requests
from datetime import datetime, timezone, timedelta
import json
import os

class NewsFilter:
    """
    Fetches the Forex Factory economic calendar and enforces
    trading embargoes around high-impact (red folder) news.
    """
    def __init__(self, embargo_minutes: int = 30):
        self.embargo_minutes = embargo_minutes
        self.cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ff_calendar_cache.json")
        self.last_fetch_date = ""
        self.events = []
        self.target_currencies = ["USD", "EUR", "CAD"]
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self._load_cache()

    def _fetch_data(self):
        """Downloads the latest weekly schedule from Forex Factory."""
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Save to cache
                with open(self.cache_file, 'w') as f:
                    json.dump({
                        "fetch_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "data": data
                    }, f)
                self.events = data
                self.last_fetch_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        except Exception as e:
            print(f"[NewsFilter] Error fetching news data: {e}")

    def _load_cache(self):
        """Loads data from cache or fetches new data if cache is stale."""
        current_date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    if cache.get("fetch_date") == current_date_utc:
                        self.events = cache.get("data", [])
                        self.last_fetch_date = current_date_utc
                    else:
                        self._fetch_data()
            except Exception:
                self._fetch_data()
        else:
            self._fetch_data()

    def is_news_embargo_active(self) -> bool:
        """
        Returns True if the current time is within +/- `embargo_minutes`
        of a High Impact news event for USD, EUR, or CAD.
        """
        # Ensure we have today's data
        current_date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.last_fetch_date != current_date_utc:
            self._fetch_data()
            
        if not self.events:
            return False # Fail open if no data
            
        now_utc = datetime.now(timezone.utc)
        embargo_delta = timedelta(minutes=self.embargo_minutes)
        
        for event in self.events:
            if event.get("impact") == "High" and event.get("country") in self.target_currencies:
                try:
                    # Date format from FF: "2026-06-19T08:30:00-04:00"
                    event_time_str = event.get("date")
                    if event_time_str:
                        event_time = datetime.fromisoformat(event_time_str)
                        # Ensure it's UTC for comparison
                        event_time_utc = event_time.astimezone(timezone.utc)
                        
                        # Check if currently inside the embargo window
                        time_diff = abs((now_utc - event_time_utc).total_seconds())
                        if time_diff <= (self.embargo_minutes * 60):
                            print(f"\n[NewsFilter] 🔴 EMBARGO ACTIVE: High Impact News '{event.get('title')}' ({event.get('country')})")
                            return True
                except Exception:
                    continue
                    
        return False

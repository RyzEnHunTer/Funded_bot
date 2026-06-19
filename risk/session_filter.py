"""
Session Filter — Block trading during low-liquidity periods.

Forex spreads widen significantly during low-volume sessions.
This filter ensures we only trade during the most liquid hours,
reducing execution costs and gap risk.
"""

from datetime import datetime, time


class SessionFilter:
    """
    Forex session-based trading filter.

    Default allowed windows (UTC):
      London:  07:00 – 16:00
      New York: 12:00 – 21:00
      Overlap:  12:00 – 16:00 (highest liquidity)

    Blocked periods:
      Friday after 20:00 UTC (weekend gap risk)
      Sunday before 22:00 UTC (thin opening liquidity)
    """

    def __init__(self,
                 london_open: time = time(7, 0),
                 london_close: time = time(16, 0),
                 ny_open: time = time(12, 0),
                 ny_close: time = time(21, 0),
                 block_friday_after: time = time(20, 0),
                 block_sunday_before: time = time(22, 0),
                 require_overlap_only: bool = True):
        """
        Parameters
        ----------
        london_open, london_close : time
            London session window (UTC).
        ny_open, ny_close : time
            New York session window (UTC).
        block_friday_after : time
            Block new trades after this time on Fridays.
        block_sunday_before : time
            Block new trades before this time on Sundays.
        require_overlap_only : bool
            If True, only trade during London/NY overlap (most liquid).
        """
        self.london_open = london_open
        self.london_close = london_close
        self.ny_open = ny_open
        self.ny_close = ny_close
        self.block_friday_after = block_friday_after
        self.block_sunday_before = block_sunday_before
        self.require_overlap_only = require_overlap_only

    def is_tradeable(self, timestamp: datetime, symbol: str = None) -> bool:
        """
        Check if the given timestamp falls within a tradeable session.

        Parameters
        ----------
        timestamp : datetime
            The bar timestamp to check. Must be UTC.
        symbol : str, optional
            The symbol being traded. Used to apply index-specific hours.).

        Returns
        -------
        bool
            True if trading is allowed at this time.
        """
        t = timestamp.time()
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday

        # Block Saturday entirely
        if weekday == 5:
            return False

        # Block Sunday before market open
        if weekday == 6 and t < self.block_sunday_before:
            return False

        # Block Friday late (weekend gap risk)
        if weekday == 4 and t >= self.block_friday_after:
            return False

        # Index-Specific Logic
        if symbol == "NQ=F":
            # US Equities Open (9:30 AM EST to 4:00 PM EST)
            # which is 13:30 to 20:00 UTC (ignoring daylight savings logic for simplicity in backtest)
            us_open = time(13, 30)
            us_close = time(20, 0)
            return us_open <= t <= us_close

        # Check session windows
        if self.require_overlap_only:
            # Only London/NY overlap
            overlap_start = max(self.london_open, self.ny_open)
            overlap_end = min(self.london_close, self.ny_close)
            return overlap_start <= t <= overlap_end

        # Allow during London OR New York session
        in_london = self.london_open <= t <= self.london_close
        in_ny = self.ny_open <= t <= self.ny_close

        return in_london or in_ny

    def get_session_name(self, timestamp: datetime, symbol: str = None) -> str:
        """Get the name of the current session."""
        t = timestamp.time()
        
        # ─── Index-Specific Logic ───
        if symbol == "NQ=F":
            # US Equities Open (9:30 AM EST to 4:00 PM EST)
            # which is 13:30 to 20:00 UTC (ignoring daylight savings logic for simplicity in backtest)
            us_open = time(13, 30)
            us_close = time(20, 0)
            return "US Equities" if us_open <= t <= us_close else "Off-Session"

        # ─── Forex Logic ───
        in_london = self.london_open <= t <= self.london_close
        in_ny = self.ny_open <= t <= self.ny_close

        if in_london and in_ny:
            return "London/NY Overlap"
        elif in_london:
            return "London"
        elif in_ny:
            return "New York"
        else:
            return "Off-Session"

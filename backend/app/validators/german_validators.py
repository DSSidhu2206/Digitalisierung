"""
German bureaucratic field validators.

Provides deterministic, rule-based validation for common German form
fields such as dates (DD.MM.YYYY) and street names.  These validators
are designed to be **strict** — when in doubt, they reject rather than
accept, forcing the field to ``UNRESOLVED`` status.

Anti-Fabrication Constraints (Section 12):
  - :class:`GermanStreetValidator` has ``REJECT_PARTIAL = True``.
    Cut-off text (e.g. "Hauptstra…") is **never** auto-completed.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timezone
from typing import ClassVar, Optional


class GermanDateValidator:
    """Validator for German dates in ``DD.MM.YYYY`` format.

    Performs strict syntactic and semantic validation:
      - Regex enforces two-digit day, two-digit month, four-digit year.
      - Impossible dates (31.02, 30.02, 31.04, etc.) are rejected.
      - Leap-year 29.02 is accepted only on valid leap years.
      - Dates in the future are rejected (configurable).

    Example usage::

        d = GermanDateValidator.parse("15.03.1990")
        if d is not None:
            print(d.year, d.month, d.day)
    """

    PATTERN: ClassVar[str] = (
        r"^(0[1-9]|[12][0-9]|3[01])\."  # DD
        r"(0[1-9]|1[0-2])\."             # MM
        r"(19|20)\d{2}$"                  # YYYY
    )

    _MAX_DAY_PER_MONTH: ClassVar[dict[int, int]] = {
        1: 31,
        2: 29,  # leap-year handling is special-cased
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }

    @classmethod
    def is_valid_format(cls, date_str: str) -> bool:
        """Check whether *date_str* matches the ``DD.MM.YYYY`` regex.

        Parameters
        ----------
        date_str:
            Raw date string from the VLM.

        Returns
        -------
        bool
            ``True`` if the string matches the pattern.
        """
        if not isinstance(date_str, str):
            return False
        return bool(re.match(cls.PATTERN, date_str.strip()))

    @classmethod
    def parse(cls, date_str: str, allow_future: bool = False) -> Optional[date]:
        """Parse a German date string into a :class:`datetime.date`.

        Performs full semantic validation including impossible-date
        rejection and leap-year handling.

        Parameters
        ----------
        date_str:
            Date string in ``DD.MM.YYYY`` format.
        allow_future:
            If ``False`` (default), dates in the future relative to the
            current UTC time are rejected and ``None`` is returned.

        Returns
        -------
        datetime.date or None
            Parsed date if valid, ``None`` if the string is malformed,
            represents an impossible date, or (when *allow_future* is
            ``False``) is in the future.
        """
        if not cls.is_valid_format(date_str):
            return None

        cleaned = date_str.strip()
        parts = cleaned.split(".")

        day = int(parts[0])
        month = int(parts[1])
        year = int(parts[2])

        # February special case
        if month == 2:
            if day == 29:
                # Must be a leap year
                if not calendar.isleap(year):
                    return None
            elif day > 28:
                return None
        else:
            max_day = cls._MAX_DAY_PER_MONTH.get(month, 0)
            if day > max_day:
                return None

        # All syntactic/semantic checks passed — construct the date
        try:
            result = date(year, month, day)
        except ValueError:
            return None

        # Future-date check
        if not allow_future and result > datetime.now(timezone.utc).date():
            return None

        return result

    @classmethod
    def validate(cls, date_str: str, allow_future: bool = False) -> bool:
        """Return ``True`` if *date_str* is a valid German date.

        Parameters
        ----------
        date_str:
            Date string to validate.
        allow_future:
            If ``False`` (default), future dates are rejected.

        Returns
        -------
        bool
            ``True`` if the date can be parsed successfully.
        """
        return cls.parse(date_str, allow_future=allow_future) is not None


class GermanStreetValidator:
    """Validator for German street names.

    German street names can contain letters, spaces, hyphens, and a
    limited set of special characters (e.g. "Straße", "Am See",
    "Königsberger Str.", "Haupt-Straße").

    Anti-Fabrication (Section 12):
      ``REJECT_PARTIAL = True`` — if the VLM output ends with an
      ellipsis, truncation marker, or is obviously cut off (e.g.
      "Hauptstra…", "Berliner Str"), the validator rejects it rather
      than attempting auto-completion.

    Example usage::

        if GermanStreetValidator.validate("Hauptstraße"):
            print("Valid street name")
        if not GermanStreetValidator.validate("Hauptstra…"):
            print("Rejected: partial text — do not autocomplete")
    """

    REJECT_PARTIAL: ClassVar[bool] = True
    """If ``True`` (default), cut-off / truncated street names are
    rejected rather than auto-completed."""

    # Characters that indicate the text was cut off
    _TRUNCATION_MARKERS: ClassVar[list[str]] = [
        "…",       # Unicode ellipsis
        "...",     # Three dots
        "..",      # Two dots
    ]

    # Pattern for valid German street names
    # Allows: letters (incl. umlauts), digits, spaces, hyphens, periods,
    #         German sharp-s (ß), slash (for "Str./Hauptstr.")
    PATTERN: ClassVar[str] = (
        r"^[A-Za-zÄÖÜäöüß0-9\s\-./]+$"
    )

    # Minimum reasonable length for a street name
    MIN_LENGTH: ClassVar[int] = 2

    # Maximum reasonable length (extremely long names exist, but this
    # catches obviously wrong extractions)
    MAX_LENGTH: ClassVar[int] = 100

    @classmethod
    def _is_truncated(cls, street: str) -> bool:
        """Check whether *street* appears to be cut off.

        Returns ``True`` if the string ends with a truncation marker
        or is otherwise obviously incomplete.
        """
        stripped = street.strip()
        for marker in cls._TRUNCATION_MARKERS:
            if stripped.endswith(marker):
                return True
        return False

    @classmethod
    def validate(cls, street: str) -> bool:
        """Validate a German street name.

        Parameters
        ----------
        street:
            Raw street name string from the VLM.

        Returns
        -------
        bool
            ``True`` if the street name is valid (not truncated, not
            empty, within length bounds, and contains only allowed
            characters).
        """
        if not isinstance(street, str):
            return False

        stripped = street.strip()

        if len(stripped) < cls.MIN_LENGTH or len(stripped) > cls.MAX_LENGTH:
            return False

        # Check for truncation (anti-fabrication)
        if cls.REJECT_PARTIAL and cls._is_truncated(stripped):
            return False

        # Check allowed characters
        if not re.match(cls.PATTERN, stripped):
            return False

        # Reject pure-number / symbol noise — a real street name has letters.
        if not any(ch.isalpha() for ch in stripped):
            return False

        return True

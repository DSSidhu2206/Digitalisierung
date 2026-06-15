"""
Checksum validation algorithms for German bureaucratic identifiers.

Implements the exact checksum algorithms used by German authorities:
  - Steueridentifikationsnummer (11-digit, prime-weighted)
  - IBAN (ISO 13616, mod-97)
  - Postleitzahl (5-digit, no checksum — length check only)

All functions are deterministic, side-effect free, and raise no
exceptions for malformed input (return ``False`` instead).
"""

from __future__ import annotations


# IBAN country code → expected length mapping (subset for EU/German context).
_IBAN_LENGTHS: dict[str, int] = {
    "DE": 22,  # Germany
    "AT": 20,  # Austria
    "CH": 21,  # Switzerland
    "NL": 18,  # Netherlands
    "BE": 16,  # Belgium
    "FR": 27,  # France
    "IT": 27,  # Italy
    "ES": 24,  # Spain
}


def validate_steuer_id(steuer_id: str) -> bool:
    """Validate a German Steueridentifikationsnummer (11 digits).

    Implements the **official** Bundeszentralamt für Steuern (BZSt)
    check-digit procedure — ISO 7064 *MOD 11,10* — together with the
    structural digit-repetition rule.  The 11th digit is the check digit,
    computed over the first ten digits.

    Structural rule (first 10 digits):
      Exactly one digit appears **two or three** times; every other digit
      appears exactly once.  The first digit must not be ``0``.

    Check digit (ISO 7064 MOD 11,10)::

        product = 10
        for d in first_ten_digits:
            s = (d + product) mod 10
            if s == 0: s = 10
            product = (s * 2) mod 11
        check = (11 - product) mod 10

    Parameters
    ----------
    steuer_id:
        The Steuer-ID string to validate.  May contain whitespace or
        common separators; these are stripped before validation.

    Returns
    -------
    bool
        ``True`` only if both the structural rule and the check digit hold.
    """
    if not isinstance(steuer_id, str):
        return False

    # Strip whitespace and common separators
    cleaned = steuer_id.replace(" ", "").replace("-", "").replace("/", "")

    # Must be exactly 11 digits
    if len(cleaned) != 11 or not cleaned.isdigit():
        return False

    digits: list[int] = [int(ch) for ch in cleaned]
    payload = digits[:10]      # first ten digits
    check_digit = digits[10]   # 11th digit is the check digit

    # The first digit of a valid IdNr is never 0.
    if payload[0] == 0:
        return False

    # Structural rule: exactly one digit repeats (2 or 3 times); rest once.
    from collections import Counter

    counts = Counter(payload)
    repeated = [d for d, c in counts.items() if c >= 2]
    if len(repeated) != 1:
        return False
    if counts[repeated[0]] not in (2, 3):
        return False
    if any(c != 1 for d, c in counts.items() if d != repeated[0]):
        return False

    # ISO 7064 MOD 11,10 check digit over the first ten digits.
    product = 10
    for d in payload:
        s = (d + product) % 10
        if s == 0:
            s = 10
        product = (s * 2) % 11
    computed = (11 - product) % 10

    return computed == check_digit


def validate_iban(iban: str) -> bool:
    """Validate an IBAN per ISO 13616 (mod-97 algorithm).

    Algorithm:
      1. Strip whitespace and convert to uppercase.
      2. Move the first four characters to the end of the string.
      3. Replace each letter with two digits (A=10, B=11, …, Z=35).
      4. Interpret the resulting string as an integer.
      5. Valid iff ``integer % 97 == 1``.

    Parameters
    ----------
    iban:
        The IBAN string to validate.  Whitespace is ignored.

    Returns
    -------
    bool
        ``True`` if the IBAN passes the mod-97 check.
    """
    if not isinstance(iban, str):
        return False

    # Remove whitespace and convert to uppercase
    cleaned = iban.replace(" ", "").replace("-", "").upper()

    if len(cleaned) < 5:
        return False

    # Basic format check: two letters followed by digits (and possibly more letters)
    if not (cleaned[0:2].isalpha() and cleaned[2:4].isdigit()):
        return False

    # Optional: validate country-specific length
    country = cleaned[0:2]
    if country in _IBAN_LENGTHS and len(cleaned) != _IBAN_LENGTHS[country]:
        return False

    # Move first 4 chars to end
    rearranged = cleaned[4:] + cleaned[:4]

    # Convert letters to digits (A=10, B=11, ..., Z=35)
    numeric = ""
    for ch in rearranged:
        if ch.isalpha():
            numeric += str(ord(ch) - ord("A") + 10)
        elif ch.isdigit():
            numeric += ch
        else:
            return False  # Invalid character

    # Mod-97 check on the numeric string
    # Handle potentially very large numbers by chunking
    remainder = 0
    # Process in chunks of at most 7 digits to stay within integer limits
    for i in range(0, len(numeric), 7):
        chunk = numeric[i : i + 7]
        remainder = int(str(remainder) + chunk) % 97

    return remainder == 1


def validate_postleitzahl(plz: str) -> bool:
    """Validate a German 5-digit postal code (Postleitzahl).

    German PLZ rules:
      - Exactly 5 decimal digits.
      - First digit must be 0-9 (any is valid in Germany).
      - Leading zeros are allowed (e.g. 01067 for Dresden).

    This is a structural check only; there is no checksum algorithm
    for German postal codes.

    Parameters
    ----------
    plz:
        The postal code string to validate.

    Returns
    -------
    bool
        ``True`` if *plz* is exactly 5 decimal digits.
    """
    if not isinstance(plz, str):
        return False

    cleaned = plz.replace(" ", "").replace("-", "")

    if len(cleaned) != 5:
        return False

    return cleaned.isdigit()

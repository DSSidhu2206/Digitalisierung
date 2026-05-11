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


# Primes used by the German Steuer-ID checksum algorithm (positions 2-10).
_STEUER_ID_PRIMES: list[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23]

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

    Algorithm (Bundesministerium der Finanzen):
      1. The string must contain exactly 11 decimal digits.
      2. Digit 1 (index 0) is the check digit.
      3. Digits 2-10 (indices 1-9) form the payload.
      4. One digit in the payload (digits 2-10) must appear **exactly**
         twice; all other payload digits must appear exactly once.
      5. Compute the weighted sum::

             sum = Σ (digit[i] * prime[i])  for i = 0..8

         where ``prime = [2, 3, 5, 7, 11, 13, 17, 19, 23]``.
      6. Check digit = 11 - (sum mod 11).  If the result is 10 the
         number is invalid.  A result of 11 is treated as 0.
      7. The computed check digit must match the first digit.

    Parameters
    ----------
    steuer_id:
        The Steuer-ID string to validate.  May contain whitespace or
        common separators; these are stripped before validation.

    Returns
    -------
    bool
        ``True`` if the Steuer-ID is structurally valid.
    """
    if not isinstance(steuer_id, str):
        return False

    # Strip whitespace and common separators
    cleaned = steuer_id.replace(" ", "").replace("-", "").replace("/", "")

    # Must be exactly 11 digits
    if len(cleaned) != 11 or not cleaned.isdigit():
        return False

    digits: list[int] = [int(ch) for ch in cleaned]
    check_digit = digits[0]
    payload = digits[1:]  # 10 digits (positions 2-11)

    # The payload (first 10 digits of the 11-digit number) must contain
    # exactly one digit that appears twice; all others appear once.
    # This means the 10-digit payload has 9 unique values.
    if len(set(payload)) != 9:
        return False

    # Count occurrences of each digit in the payload
    from collections import Counter

    counts = Counter(payload)
    twice = [d for d, c in counts.items() if c == 2]
    if len(twice) != 1:
        return False

    # Compute weighted sum over the first 9 payload digits
    # (positions 2-10, i.e. indices 0-8 of payload)
    weighted_sum = sum(d * p for d, p in zip(payload[:9], _STEUER_ID_PRIMES))

    # Check digit calculation
    remainder = weighted_sum % 11
    computed = 11 - remainder

    if computed == 10:
        return False  # 10 is an invalid check-digit result
    if computed == 11:
        computed = 0

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

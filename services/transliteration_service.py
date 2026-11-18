"""Offline Hindi-to-Roman caption transliteration with timestamp preservation."""

from __future__ import annotations

import copy
import re
from typing import Any


_DEVANAGARI = re.compile(r"[\u0900-\u097F]+")
_TOKEN = re.compile(r"([\u0900-\u097F]+|[^\u0900-\u097F]+)")

_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee",
    "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e",
    "ऐ": "ai", "ओ": "o", "औ": "au",
}

_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u",
    "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au",
}

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z",
    "ड़": "r", "ढ़": "rh", "फ़": "f",
}

_COMMON_WORDS = {
    "आज": "aaj", "अभी": "abhi", "अगर": "agar",
    "अच्छा": "achha", "आसान": "aasaan", "आप": "aap",
    "आपका": "aapka", "आपकी": "aapki", "आपके": "aapke",
    "इंसान": "insaan", "इस": "is", "इसका": "iska",
    "इसकी": "iski", "इसलिए": "isliye", "उनका": "unka",
    "उनकी": "unki", "एक": "ek", "और": "aur", "कब": "kab",
    "कहाँ": "kahaan", "कहानी": "kahaani", "क्यों": "kyun",
    "क्या": "kya", "कैसे": "kaise", "कोई": "koi",
    "कर": "kar", "करना": "karna", "करते": "karte",
    "करती": "karti", "करता": "karta", "किया": "kiya",
    "किस": "kis", "किसी": "kisi", "की": "ki", "के": "ke",
    "को": "ko", "खुद": "khud", "गया": "gaya", "गई": "gayi",
    "घर": "ghar", "जब": "jab", "जहाँ": "jahaan",
    "जैसे": "jaise", "जो": "jo", "तक": "tak", "तब": "tab",
    "तुम": "tum", "तुम्हारा": "tumhara",
    "तुम्हारी": "tumhari", "तो": "to", "था": "tha",
    "थी": "thi", "थे": "the", "दिया": "diya",
    "दुनिया": "duniya", "नहीं": "nahi", "नई": "nayi",
    "ने": "ne", "पर": "par", "पहले": "pehle",
    "पानी": "paani", "फिर": "phir", "बहुत": "bahut",
    "बात": "baat", "बिना": "bina", "भी": "bhi",
    "मुझे": "mujhe", "मेरा": "mera", "मेरी": "meri",
    "मेरे": "mere", "में": "mein", "यह": "yeh",
    "यहाँ": "yahaan", "या": "ya", "रहा": "raha",
    "रही": "rahi", "रहे": "rahe", "लिए": "liye",
    "लेकिन": "lekin", "लोग": "log", "वह": "woh",
    "वहाँ": "wahaan", "सब": "sab", "सकता": "sakta",
    "सकती": "sakti", "सकते": "sakte", "साथ": "saath",
    "सामने": "saamne", "से": "se", "हमें": "humein",
    "हम": "hum", "हमारा": "hamara", "हमारी": "hamari",
    "हैं": "hain", "है": "hai", "हो": "ho",
    "होगा": "hoga", "होगी": "hogi", "होगे": "hoge",
}


def _transliterate_word(word: str) -> str:
    """Transliterate one Hindi token using offline rules."""
    if word in _COMMON_WORDS:
        return _COMMON_WORDS[word]

    output: list[str] = []
    index = 0

    while index < len(word):
        char = word[index]

        if char in _CONSONANTS:
            output.append(_CONSONANTS[char])
            next_index = index + 1

            if next_index < len(word) and word[next_index] in _MATRAS:
                output.append(_MATRAS[word[next_index]])
                index += 1
            elif next_index < len(word) and word[next_index] == "्":
                index += 1
            else:
                output.append("a")

        elif char in _VOWELS:
            output.append(_VOWELS[char])
        elif char in {"ं", "ँ"}:
            output.append("n")
        elif char == "ः":
            output.append("h")
        elif char in {"़", "्"}:
            pass
        elif char == "।":
            output.append(".")
        elif char == "॥":
            output.append("..")
        elif char.isdigit():
            output.append(char)
        else:
            output.append(char)

        index += 1

    result = "".join(output)

    # Approximate Hindi word-final inherent-vowel deletion.
    if result.endswith("a") and len(result) > 1:
        result = result[:-1]

    return result


def transliterate_text(text: str) -> str:
    """Romanize Devanagari text without translating its meaning."""
    if not isinstance(text, str) or not text:
        return text

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        converted = _transliterate_word(token)

        if token.istitle():
            return converted[:1].upper() + converted[1:]
        return converted

    return _DEVANAGARI.sub(replace, text)


class TransliterationService:
    """Prepare display captions without modifying the original transcript."""

    def prepare(
        self,
        transcript: dict[str, Any],
        settings: Any,
    ) -> dict[str, Any]:
        prepared = copy.deepcopy(transcript)
        language = str(prepared.get("language", "unknown")).lower()
        enabled = bool(getattr(settings, "hindi_caption_romanize", True))
        is_hindi = language in {"hi", "hindi"}

        for segment in prepared.get("segments", []):
            for word in segment.get("words", []):
                original = str(word.get("word", ""))
                word["display_word"] = (
                    transliterate_text(original)
                    if enabled and is_hindi
                    else original
                )

            original_text = str(segment.get("text", ""))
            segment["display_text"] = (
                transliterate_text(original_text)
                if enabled and is_hindi
                else original_text
            )

        return prepared

    def apply_to_scenes(
        self,
        transcript: dict[str, Any],
        scenes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach timestamp-aligned display captions to scene copies."""
        prepared_scenes = copy.deepcopy(scenes)
        segments = transcript.get("segments", [])

        for scene in prepared_scenes:
            start = float(scene["start"])
            end = float(scene["end"])
            display_parts: list[str] = []

            for segment in segments:
                segment_start = float(segment.get("start", 0))
                segment_end = float(segment.get("end", 0))

                if segment_end <= start or segment_start >= end:
                    continue

                words = segment.get("words") or []

                if words:
                    for word in words:
                        word_start = float(word.get("start", segment_start))
                        word_end = float(word.get("end", segment_end))

                        if word_end > start and word_start < end:
                            display_word = str(
                                word.get("display_word", word.get("word", ""))
                            ).strip()
                            if display_word:
                                display_parts.append(display_word)
                else:
                    display_text = str(
                        segment.get("display_text", segment.get("text", ""))
                    ).strip()
                    if display_text:
                        display_parts.append(display_text)

            caption = scene.setdefault("caption", {})
            caption["display_text"] = " ".join(display_parts).strip()

        return prepared_scenes

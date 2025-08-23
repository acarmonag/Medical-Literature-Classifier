# src/utils/text_cleaning.py
# Purpose: Robust medical text cleaner with configurable rules for citations, quotes, digits, and unicode normalization.

from __future__ import annotations
import re
import unicodedata
from html import unescape
from typing import Iterable, List
from sklearn.base import BaseEstimator, TransformerMixin

class MedTextCleaner(BaseEstimator, TransformerMixin):
    """
    Clean medical titles/abstracts with configurable rules.

    Key features:
    - Drop bracketed/parenthetical citations and figure/table mentions.
    - Strip quotes (straight + curly) and miscellaneous symbols.
    - Control digits removal with preservation for biomedical hyphen-numeric tokens (e.g., 'covid-19', 'il-6').
    - Unicode normalization (NFKC) and whitespace compaction.
    """

    def __init__(
        self,
        drop_citations: bool = True,
        strip_quotes: bool = True,
        remove_digits: str = "standalone",  # 'none' | 'standalone' | 'all'
        preserve_hyphen_numbers: bool = True,
        lower: bool = True,
        drop_urls_emails: bool = True,
        map_greek_letters: bool = True,
    ):
        self.drop_citations = drop_citations
        self.strip_quotes = strip_quotes
        self.remove_digits = remove_digits
        self.preserve_hyphen_numbers = preserve_hyphen_numbers
        self.lower = lower
        self.drop_urls_emails = drop_urls_emails
        self.map_greek_letters = map_greek_letters

        # Compile heavy regexes once
        # [1], [12,13], [doi:...], [PMID: 12345]
        self._rx_square_cites = re.compile(r"\[(?:\s*\d+(?:\s*[,;–-]\s*\d+)*\s*|[^]]{1,80})\]")
        # (2020), (n=43), (p<0.05), (Smith et al., 2020)
        self._rx_paren_cites = re.compile(
            r"\((?:\s*(?:\d{4}[a-z]?|n\s*=\s*\d+|ci\s*\d+%|p\s*[<=>]\s*\d*\.?\d+|[A-Z][A-Za-z]+(?:\s+et al\.)?,?\s*\d{4}[a-z]?)\s*)\)",
            flags=re.IGNORECASE,
        )
        # Fig./Figure/Table/Supplementary + number
        self._rx_fig_table = re.compile(r"\b(?:fig(?:ure)?|table|supplementary|supp)\s*[.\-]?\s*\d+\b", re.IGNORECASE)
        # URLs & emails
        self._rx_url = re.compile(r"https?://\S+|www\.\S+")
        self._rx_email = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
        # Quotes (straight + curly)
        self._rx_quotes = re.compile(r'["“”‘’\'`´]+')
        # Non-word symbols except whitespace and hyphen (we keep hyphen to preserve tokens like covid-19)
        self._rx_symbols = re.compile(r"[^\w\s\-]+", re.UNICODE)
        # Standalone small numbers (1-3 digits) – for aggressive cleanup of references like "Table 3"
        self._rx_small_digits = re.compile(r"\b\d{1,3}\b")
        # Biomedical hyphen-numeric tokens to preserve: word(s)-digits or digits-word(s)
        self._rx_bio_hyphen_digits = re.compile(r"\b(?:[A-Za-z]+(?:-[A-Za-z]+)*-\d+[A-Za-z]*|\d+[A-Za-z]*(?:-[A-Za-z]+)+)\b")

        # Greek letter mapping (common in gene/cytokine names)
        self._greek_map = {
            "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
            "κ": "kappa", "λ": "lambda", "μ": "mu", "π": "pi", "σ": "sigma", "τ": "tau",
            "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
        }

    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]) -> List[str]:
        return [self._clean_one(text) for text in X]

    def _clean_one(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # HTML/unicode normalize
        s = unescape(text)
        s = unicodedata.normalize("NFKC", s)

        if self.lower:
            s = s.lower()

        if self.drop_urls_emails:
            s = self._rx_url.sub(" ", s)
            s = self._rx_email.sub(" ", s)

        if self.drop_citations:
            s = self._rx_square_cites.sub(" ", s)
            s = self._rx_paren_cites.sub(" ", s)
            s = self._rx_fig_table.sub(" ", s)

        if self.strip_quotes:
            s = self._rx_quotes.sub(" ", s)

        if self.map_greek_letters:
            for g, latin in self._greek_map.items():
                s = s.replace(g, f" {latin} ")

        # Remove miscellaneous symbols while keeping hyphens for biomedical tokens
        s = self._rx_symbols.sub(" ", s)

        # Control digits removal with preservation of biomedical hyphenated tokens
        if self.remove_digits != "none":
            if self.preserve_hyphen_numbers:
                preserved = {}
                def _mask(m):
                    key = f"__PRESERVE_{len(preserved)}__"
                    preserved[key] = m.group(0)
                    return key
                s = self._rx_bio_hyphen_digits.sub(_mask, s)

            if self.remove_digits == "standalone":
                s = self._rx_small_digits.sub(" ", s)
            elif self.remove_digits == "all":
                s = re.sub(r"\d+", " ", s)

            if self.preserve_hyphen_numbers:
                for key, token in preserved.items():
                    s = s.replace(key, token)

        # Collapse spaces
        s = re.sub(r"\s+", " ", s).strip()
        return s
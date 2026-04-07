"""Modular indicator package."""

from .base import Indicator, IndicatorResult, IndicatorRunResult
from .registry import (
    INDICATOR_REGISTRY,
    attach_indicators,
    build_indicator,
    build_indicators,
    register_indicator,
    run_indicators,
)
from .rsi import RSI
from .macd import MACD
from .bollinger_bands import BollingerBands
from .atr import ATR
from .fair_value_gap import FairValueGap


__all__ = [
    "INDICATOR_REGISTRY",
    "ATR",
    "BollingerBands",
    "FairValueGap",
    "Indicator",
    "IndicatorResult",
    "IndicatorRunResult",
    "MACD",
    "RSI",
    "attach_indicators",
    "build_indicator",
    "build_indicators",
    "register_indicator",
    "run_indicators",
]
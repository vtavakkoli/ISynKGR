"""Backward-compatible hybrid module facade.

Tests and legacy callers monkeypatch symbols from this module directly.
Keep this facade stateful so monkeypatches continue to affect pipeline execution.
"""

from __future__ import annotations

from typing import Any

import isynkgr.pipeline.adaptive_candidate_ranker as _acr

TranslatorConfig = _acr.TranslatorConfig
Mode = _acr.Mode
AdaptiveCandidateRankerPipeline = _acr.AdaptiveCandidateRankerPipeline

# Backward-compatible mutable alias that tests monkeypatch.
ADAPTERS = _acr.ADAPTERS


class HybridPipeline(_acr.AdaptiveCandidateRankerPipeline):
    def run(self, *args: Any, **kwargs: Any):
        # Keep adaptive module in sync with monkeypatched hybrid-module globals.
        _acr.ADAPTERS = ADAPTERS
        return super().run(*args, **kwargs)


__all__ = ["AdaptiveCandidateRankerPipeline", "HybridPipeline", "TranslatorConfig", "Mode", "ADAPTERS"]

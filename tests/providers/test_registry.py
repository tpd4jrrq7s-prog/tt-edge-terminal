"""Tests for the explicit, non-global provider registry."""

from __future__ import annotations

import pytest

from providers.errors import ProviderNotRegisteredError
from providers.generic.adapter import GenericProviderAdapter
from providers.generic.mappings import mock_provider_mapping
from providers.registry import ProviderRegistry


def test_register_and_get():
    registry = ProviderRegistry()
    adapter = GenericProviderAdapter(mock_provider_mapping())
    registry.register(adapter)
    assert registry.get("mock") is adapter


def test_unregistered_provider_raises():
    registry = ProviderRegistry()
    with pytest.raises(ProviderNotRegisteredError):
        registry.get("nope")


def test_providers_lists_sorted_names():
    registry = ProviderRegistry()
    registry.register(GenericProviderAdapter(mock_provider_mapping()))
    assert registry.providers() == ["mock"]


def test_registry_instances_are_independent():
    a = ProviderRegistry()
    b = ProviderRegistry()
    a.register(GenericProviderAdapter(mock_provider_mapping()))
    assert a.providers() == ["mock"]
    assert b.providers() == []

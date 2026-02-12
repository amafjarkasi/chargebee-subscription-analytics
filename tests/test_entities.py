"""Unit tests for chargebee_mapper.entities module."""

import pytest

from chargebee_mapper.entities import (
    EntityDef,
    INDEPENDENT_ENTITIES,
    DEPENDENT_ENTITIES,
    ALL_ENTITIES,
)


class TestEntityDef:
    """Tests for EntityDef dataclass."""

    def test_basic_entity(self):
        """Test basic EntityDef creation."""
        entity = EntityDef(
            name="Customer",
            key="customers",
            sdk_resource="Customer",
            response_key="customer",
        )
        
        assert entity.name == "Customer"
        assert entity.key == "customers"
        assert entity.sdk_resource == "Customer"
        assert entity.response_key == "customer"
        assert entity.requires_parent is False
        assert entity.parent_entity == ""
        assert entity.parent_id_field == ""
        assert entity.no_list_params is False

    def test_dependent_entity(self):
        """Test EntityDef with parent dependency."""
        entity = EntityDef(
            name="Attached Item",
            key="attached_items",
            sdk_resource="AttachedItem",
            response_key="attached_item",
            requires_parent=True,
            parent_entity="items",
            parent_id_field="id",
        )
        
        assert entity.requires_parent is True
        assert entity.parent_entity == "items"

    def test_no_list_params_entity(self):
        """Test EntityDef with no_list_params flag."""
        entity = EntityDef(
            name="Configuration",
            key="configurations",
            sdk_resource="Configuration",
            response_key="configuration",
            no_list_params=True,
        )
        
        assert entity.no_list_params is True

    def test_entity_is_frozen(self):
        """Test that EntityDef is immutable (frozen dataclass)."""
        entity = EntityDef(
            name="Customer",
            key="customers",
            sdk_resource="Customer",
            response_key="customer",
        )
        
        with pytest.raises(AttributeError):
            entity.name = "Modified"


class TestEntityLists:
    """Tests for entity list definitions."""

    def test_independent_entities_not_empty(self):
        """Test that INDEPENDENT_ENTITIES is populated."""
        assert len(INDEPENDENT_ENTITIES) > 0

    def test_dependent_entities_exist(self):
        """Test that DEPENDENT_ENTITIES contains at least one entity."""
        assert len(DEPENDENT_ENTITIES) >= 1

    def test_all_entities_is_combination(self):
        """Test that ALL_ENTITIES combines independent and dependent."""
        assert len(ALL_ENTITIES) == len(INDEPENDENT_ENTITIES) + len(DEPENDENT_ENTITIES)

    def test_customer_entity_exists(self):
        """Test that Customer entity is defined."""
        customer = next((e for e in ALL_ENTITIES if e.key == "customers"), None)
        assert customer is not None
        assert customer.name == "Customer"
        assert customer.sdk_resource == "Customer"

    def test_subscription_entity_exists(self):
        """Test that Subscription entity is defined."""
        sub = next((e for e in ALL_ENTITIES if e.key == "subscriptions"), None)
        assert sub is not None
        assert sub.name == "Subscription"

    def test_invoice_entity_exists(self):
        """Test that Invoice entity is defined."""
        invoice = next((e for e in ALL_ENTITIES if e.key == "invoices"), None)
        assert invoice is not None
        assert invoice.name == "Invoice"

    def test_attached_item_is_dependent(self):
        """Test that AttachedItem is in DEPENDENT_ENTITIES."""
        attached = next((e for e in DEPENDENT_ENTITIES if e.key == "attached_items"), None)
        assert attached is not None
        assert attached.requires_parent is True
        assert attached.parent_entity == "items"

    def test_all_entities_have_unique_keys(self):
        """Test that all entity keys are unique."""
        keys = [e.key for e in ALL_ENTITIES]
        assert len(keys) == len(set(keys)), "Duplicate entity keys found"

    def test_all_entities_have_required_fields(self):
        """Test that all entities have required fields populated."""
        for entity in ALL_ENTITIES:
            assert entity.name, f"Entity missing name: {entity}"
            assert entity.key, f"Entity missing key: {entity}"
            assert entity.sdk_resource, f"Entity missing sdk_resource: {entity}"
            assert entity.response_key, f"Entity missing response_key: {entity}"

    def test_dependent_entities_have_parent_info(self):
        """Test that dependent entities have parent information."""
        for entity in DEPENDENT_ENTITIES:
            assert entity.requires_parent is True, f"{entity.name} should require parent"
            assert entity.parent_entity, f"{entity.name} missing parent_entity"

    def test_entity_count(self):
        """Test expected number of entities."""
        # Based on the entities.py file, we expect 36 total entities
        assert len(ALL_ENTITIES) == 36
        assert len(INDEPENDENT_ENTITIES) == 35
        assert len(DEPENDENT_ENTITIES) == 1

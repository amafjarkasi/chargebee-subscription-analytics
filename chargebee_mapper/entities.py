"""Entity registry defining all Chargebee API v2 resources to fetch.

CHARGEBEE API v2 - COMPLETE ENTITY RELATIONSHIP MAP
====================================================

Customer (root entity)
  +-- Subscription (customer_id -> Customer.id)
  |     +-- Invoice (subscription_id -> Subscription.id)
  |     |     +-- Credit Note (reference_invoice_id -> Invoice.id)
  |     |     +-- Transaction (invoice_id -> Invoice.id, linked_payments)
  |     |     +-- Comment (entity_type=invoice, entity_id -> Invoice.id)
  |     +-- Unbilled Charge (subscription_id -> Subscription.id)
  |     +-- Quote (subscription_id -> Subscription.id)
  |     +-- Order (subscription_id -> Subscription.id)
  |     +-- Gift (subscription_id -> Subscription.id)
  |     +-- Ramp (subscription_id -> Subscription.id)
  |     +-- Attached Item (via Item -> Subscription)
  +-- Payment Source (customer_id -> Customer.id)
  +-- Promotional Credit (customer_id -> Customer.id)
  +-- Contact (nested in Customer response)
  +-- Comment (entity_type=customer, entity_id -> Customer.id)

Product Catalog
  +-- Item Family (grouping)
  |     +-- Item (item_family_id -> ItemFamily.id)
  |           +-- Item Price (item_id -> Item.id)
  |           |     +-- Differential Price (item_price_id -> ItemPrice.id)
  |           +-- Attached Item (item_id -> Item.id) [DEPENDENT - requires parent]
  +-- Plan (legacy, product_catalog_version=1 only)
  +-- Addon (legacy, product_catalog_version=1 only)
  +-- Price Variant (feature-gated, may not be enabled on all sites)

Discounts & Promotions
  +-- Coupon (standalone discount definition)
  |     +-- Coupon Set (coupon_id -> Coupon.id, groups of codes)
  |     +-- Coupon Code (coupon_set_id -> CouponSet.id, individual codes)
  +-- Promotional Credit (customer_id -> Customer.id)

System & Configuration
  +-- Event (audit log of all changes across entities)
  +-- Comment (polymorphic, on any entity via entity_type + entity_id)
  +-- Hosted Page (checkout/portal/manage payment pages)
  +-- Currency (site-level currency configuration)
  +-- Configuration (site-level settings; no ListParams, simple list)
  +-- Feature (entitlement feature definitions)
  |     +-- Entitlement (feature_id -> Feature.id)
  +-- Site Migration Detail (cross-site migration tracking)
  +-- Webhook Endpoint (webhook URL configurations)
  +-- Usage (usage-based billing records)

Omnichannel (mobile app store integrations)
  +-- Omnichannel Subscription (Apple/Google store subscriptions)
  +-- Omnichannel One-Time Order (Apple/Google store purchases)

NON-LISTABLE RESOURCES (63 total, accessed via parent or specific endpoints):
  Address, AdvanceInvoiceSchedule, Attribute, BillingConfiguration, Brand,
  BusinessEntity, BusinessEntityTransfer, Card, Contact, ContractTerm,
  CreditNoteEstimate, CustomerEntitlement, Discount, Download, Einvoice,
  EntitlementOverride, Estimate, Export, GatewayErrorDetail, Hierarchy,
  ImpactedCustomer, ImpactedItem, ImpactedItemPrice, ImpactedSubscription,
  InAppSubscription, InvoiceEstimate, ItemEntitlement, Metadata, OfferEvent,
  OfferFulfillment, OmnichannelOneTimeOrderItem, OmnichannelSubscriptionItem,
  OmnichannelSubscriptionItemOffer, OmnichannelSubscriptionItemScheduledChange,
  OmnichannelTransaction, PaymentIntent, PaymentReferenceNumber,
  PaymentSchedule, PaymentScheduleEstimate, PaymentScheduleScheme,
  PaymentVoucher, PersonalizedOffer, PortalSession, PricingPageSession,
  Purchase, QuoteLineGroup, QuotedCharge, QuotedDeltaRamp, QuotedRamp,
  QuotedSubscription, RecordedPurchase, ResourceMigration, Rule,
  SubscriptionEntitlement, SubscriptionEntitlementsCreatedDetail,
  SubscriptionEntitlementsUpdatedDetail, SubscriptionEstimate,
  TaxWithheld, ThirdPartyPaymentMethod, TimeMachine, Token,
  UsageEvent, UsageFile

KEY DATA POINTS PER ENTITY
==========================

Customer:
  id, first_name, last_name, email, phone, company, auto_collection,
  net_term_days, allow_direct_debit, created_at, updated_at, locale,
  taxability, vat_number, billing_address, card_status, channel,
  resource_version, deleted, promotional_credits, refundable_credits,
  excess_payments, unbilled_charges, preferred_currency_code,
  business_entity_id, cf_* (custom fields)

Subscription:
  id, customer_id, plan_id, plan_quantity, plan_unit_price, status,
  trial_start, trial_end, current_term_start, current_term_end,
  next_billing_at, created_at, started_at, activated_at, cancelled_at,
  cancel_reason, remaining_billing_cycles, subscription_items[],
  item_tiers[], charged_items[], coupons[], shipping_address,
  payment_source_id, auto_collection, due_invoices_count, channel,
  resource_version, deleted, has_scheduled_changes, business_entity_id

Invoice:
  id, customer_id, subscription_id, recurring, status, price_type,
  date, due_date, total, amount_paid, amount_adjusted, amount_due,
  credits_applied, write_off_amount, sub_total, tax, first_invoice,
  currency_code, line_items[], discounts[], taxes[], linked_payments[],
  linked_orders[], notes[], shipping_address, billing_address,
  dunning_status, payment_owner, round_off_amount, channel,
  resource_version, deleted, generated_at, updated_at

Credit Note:
  id, customer_id, subscription_id, reference_invoice_id, type, reason_code,
  status, date, total, amount_allocated, amount_refunded, amount_available,
  price_type, currency_code, line_items[], discounts[], taxes[],
  linked_refunds[], allocations[], resource_version, deleted, updated_at

Transaction:
  id, customer_id, subscription_id, gateway_account_id, payment_source_id,
  payment_method, gateway, type, date, settled_at, exchange_rate, amount,
  currency_code, status, fraud_flag, authorization_reason,
  linked_invoices[], linked_credit_notes[], linked_refunds[],
  resource_version, deleted, updated_at

Order:
  id, document_number, invoice_id, subscription_id, customer_id, status,
  cancellation_reason, payment_status, order_type, price_type,
  order_date, shipping_date, created_at, updated_at, order_line_items[],
  shipping_address, billing_address, resource_version, deleted

Quote:
  id, name, customer_id, subscription_id, invoice_id, status, operation_type,
  vat_number, date, total_payable, total, charge_on_acceptance,
  currency_code, line_items[], discounts[], taxes[], line_item_discounts[],
  line_item_taxes[], line_item_tiers[], resource_version, updated_at

Gift:
  id, status, gift_receiver, gift_timeline, gifter, scheduled_at,
  auto_claim, no_expiry, claim_expiry_date, resource_version, updated_at

Item Family:
  id, name, description, status, channel, resource_version, updated_at

Item:
  id, name, description, status, type, item_family_id,
  item_applicability, is_shippable, is_giftable, enabled_for_checkout,
  enabled_in_portal, redirect_url, metered, channel,
  resource_version, updated_at

Item Price:
  id, name, item_id, item_family_id, item_type, description,
  status, pricing_model, price, period, period_unit, trial_period,
  trial_period_unit, currency_code, free_quantity, channel,
  resource_version, updated_at, tiers[], tax_detail

Differential Price:
  id, item_price_id, parent_item_id, price, status,
  currency_code, tiers[], resource_version, updated_at

Price Variant:
  id, name, description, status, created_at, updated_at,
  resource_version, attributes[]

Attached Item:
  id, item_id, type, status, quantity, billing_cycles,
  charge_on_event, charge_once, created_at, updated_at, resource_version

Plan (legacy catalog v1 only):
  id, name, invoice_name, description, price, period, period_unit,
  trial_period, trial_period_unit, pricing_model, charge_model, status,
  enabled_in_hosted_pages, enabled_in_portal, addon_applicability,
  currency_code, taxable, tax_profile_id, tiers[], resource_version

Addon (legacy catalog v1 only):
  id, name, invoice_name, description, pricing_model, charge_type,
  price, period, period_unit, unit, status, currency_code, type,
  enabled_in_portal, taxable, tiers[], resource_version

Coupon:
  id, name, invoice_name, discount_type, discount_amount,
  discount_percentage, currency_code, duration_type, duration_month,
  max_redemptions, status, apply_on, plan_constraint, addon_constraint,
  valid_till, created_at, resource_version, updated_at

Coupon Set:
  id, coupon_id, name, total_count, redeemed_count, archived_count,
  resource_version, updated_at

Coupon Code:
  code, coupon_id, coupon_set_id, status, resource_version, updated_at

Payment Source:
  id, customer_id, type, status, gateway, gateway_account_id,
  reference_id, ip_address, issuing_country, created_at, updated_at,
  card, bank_account, amazon_payment, paypal_express_checkout,
  resource_version, deleted

Virtual Bank Account:
  id, customer_id, email, scheme, bank_name, account_number,
  routing_number, swift_code, gateway, gateway_account_id,
  reference_id, resource_version, updated_at, deleted

Unbilled Charge:
  id, customer_id, subscription_id, date_from, date_to, unit_amount,
  pricing_model, quantity, amount, currency_code, discount_amount,
  description, entity_type, entity_id, is_voided, voided_at,
  resource_version, deleted, updated_at

Promotional Credit:
  id, customer_id, type, amount, currency_code, description,
  credit_type, closing_balance, created_at, resource_version

Usage:
  id, subscription_id, item_price_id, quantity, usage_date,
  source, note, resource_version, updated_at

Event:
  id, occurred_at, source, user, event_type, api_version,
  content (embedded entity snapshot), webhooks[], resource_version

Comment:
  id, entity_type, entity_id, added_by, notes, created_at,
  type, resource_version

Hosted Page:
  id, type, url, state, pass_thru_content, created_at, expires_at,
  updated_at, resource_version, content (checkout result)

Feature:
  id, name, description, status, type, levels[],
  resource_version, updated_at

Entitlement:
  id, entity_id, entity_type, feature_id, feature_name, value,
  resource_version, updated_at

Currency:
  id, enabled, forex_type, currency_code, is_base_currency,
  manual_exchange_rate, resource_version

Configuration:
  (Site-level configuration; no documented fields -- returns raw config)

Site Migration Detail:
  entity_id, entity_id_at_other_site, entity_type, status,
  other_site_name, resource_version, updated_at

Ramp:
  id, description, subscription_id, status, effective_from,
  created_at, updated_at, items_to_add[], items_to_update[],
  items_to_remove[], coupons_to_add[], coupons_to_remove[],
  discounts_to_add[], discounts_to_remove[], resource_version

Webhook Endpoint:
  id, url, api_version, include_auto_close_events

Omnichannel Subscription:
  id, app_id, store, initial_purchase_transaction_id

Omnichannel One-Time Order:
  id, app_id, store, initial_purchase_transaction_id
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EntityDef:
    name: str  # Display name
    key: str  # snake_case key for files/tables
    sdk_resource: str  # Chargebee SDK resource class name
    response_key: str  # Key in response list entries
    requires_parent: bool = False  # Whether list() needs a parent ID as first arg
    parent_entity: str = ""  # Key of parent entity whose IDs we iterate over
    parent_id_field: str = ""  # (unused for positional-arg parents, kept for docs)
    no_list_params: bool = False  # True if list() takes no ListParams (e.g. Configuration)


# Independent entities - can be fetched with resource.list(params)
INDEPENDENT_ENTITIES: list[EntityDef] = [
    # Core Business
    EntityDef(name="Customer", key="customers", sdk_resource="Customer", response_key="customer"),
    EntityDef(name="Subscription", key="subscriptions", sdk_resource="Subscription", response_key="subscription"),
    EntityDef(name="Invoice", key="invoices", sdk_resource="Invoice", response_key="invoice"),
    EntityDef(name="Credit Note", key="credit_notes", sdk_resource="CreditNote", response_key="credit_note"),
    EntityDef(name="Transaction", key="transactions", sdk_resource="Transaction", response_key="transaction"),
    EntityDef(name="Order", key="orders", sdk_resource="Order", response_key="order"),
    EntityDef(name="Quote", key="quotes", sdk_resource="Quote", response_key="quote"),
    EntityDef(name="Gift", key="gifts", sdk_resource="Gift", response_key="gift"),
    # Product Catalog
    EntityDef(name="Item Family", key="item_families", sdk_resource="ItemFamily", response_key="item_family"),
    EntityDef(name="Item", key="items", sdk_resource="Item", response_key="item"),
    EntityDef(name="Item Price", key="item_prices", sdk_resource="ItemPrice", response_key="item_price"),
    EntityDef(name="Differential Price", key="differential_prices", sdk_resource="DifferentialPrice", response_key="differential_price"),
    EntityDef(name="Price Variant", key="price_variants", sdk_resource="PriceVariant", response_key="price_variant"),
    EntityDef(name="Plan", key="plans", sdk_resource="Plan", response_key="plan"),
    EntityDef(name="Addon", key="addons", sdk_resource="Addon", response_key="addon"),
    EntityDef(name="Coupon", key="coupons", sdk_resource="Coupon", response_key="coupon"),
    EntityDef(name="Coupon Set", key="coupon_sets", sdk_resource="CouponSet", response_key="coupon_set"),
    EntityDef(name="Coupon Code", key="coupon_codes", sdk_resource="CouponCode", response_key="coupon_code"),
    # Payments & Billing
    EntityDef(name="Payment Source", key="payment_sources", sdk_resource="PaymentSource", response_key="payment_source"),
    EntityDef(name="Virtual Bank Account", key="virtual_bank_accounts", sdk_resource="VirtualBankAccount", response_key="virtual_bank_account"),
    EntityDef(name="Unbilled Charge", key="unbilled_charges", sdk_resource="UnbilledCharge", response_key="unbilled_charge"),
    EntityDef(name="Promotional Credit", key="promotional_credits", sdk_resource="PromotionalCredit", response_key="promotional_credit"),
    EntityDef(name="Usage", key="usages", sdk_resource="Usage", response_key="usage"),
    # System & Config
    EntityDef(name="Event", key="events", sdk_resource="Event", response_key="event"),
    EntityDef(name="Comment", key="comments", sdk_resource="Comment", response_key="comment"),
    EntityDef(name="Hosted Page", key="hosted_pages", sdk_resource="HostedPage", response_key="hosted_page"),
    EntityDef(name="Feature", key="features", sdk_resource="Feature", response_key="feature"),
    EntityDef(name="Entitlement", key="entitlements", sdk_resource="Entitlement", response_key="entitlement"),
    EntityDef(name="Currency", key="currencies", sdk_resource="Currency", response_key="currency"),
    EntityDef(name="Configuration", key="configurations", sdk_resource="Configuration", response_key="configuration", no_list_params=True),
    EntityDef(name="Site Migration Detail", key="site_migration_details", sdk_resource="SiteMigrationDetail", response_key="site_migration_detail"),
    EntityDef(name="Ramp", key="ramps", sdk_resource="Ramp", response_key="ramp"),
    EntityDef(name="Webhook Endpoint", key="webhook_endpoints", sdk_resource="WebhookEndpoint", response_key="webhook_endpoint"),
    # Omnichannel
    EntityDef(name="Omnichannel Subscription", key="omnichannel_subscriptions", sdk_resource="OmnichannelSubscription", response_key="omnichannel_subscription"),
    EntityDef(name="Omnichannel One-Time Order", key="omnichannel_one_time_orders", sdk_resource="OmnichannelOneTimeOrder", response_key="omnichannel_one_time_order"),
]

# Dependent entities - list() takes a parent ID as the first positional argument
DEPENDENT_ENTITIES: list[EntityDef] = [
    EntityDef(
        name="Attached Item",
        key="attached_items",
        sdk_resource="AttachedItem",
        response_key="attached_item",
        requires_parent=True,
        parent_entity="items",
        parent_id_field="id",
    ),
]

ALL_ENTITIES = INDEPENDENT_ENTITIES + DEPENDENT_ENTITIES

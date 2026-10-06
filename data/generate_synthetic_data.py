"""Generate deterministic synthetic e-commerce data for this portfolio case study.

The records are entirely fictional and must not be interpreted as company results.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_CUSTOMERS = 4_000
N_PRODUCTS = 180
START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2025-12-31")
OUTPUT_DIR = Path(__file__).resolve().parent / "sample"


def generate_products(rng: np.random.Generator) -> pd.DataFrame:
    """Create a catalog with category-specific price levels.

    Products are deliberately unevenly distributed across categories so the
    later category-affinity recommender has realistic choices to rank.
    """
    categories = {
        "Electronics": 85,
        "Home": 42,
        "Beauty": 24,
        "Sports": 38,
        "Books": 16,
        "Fashion": 32,
        "Grocery": 12,
        "Pet Care": 20,
    }
    category_names = list(categories)
    category_probs = np.array([0.12, 0.16, 0.13, 0.12, 0.12, 0.15, 0.12, 0.08])
    chosen_categories = rng.choice(category_names, size=N_PRODUCTS, p=category_probs)
    category_counter = {category: 0 for category in category_names}
    rows = []
    for product_number, category in enumerate(chosen_categories, start=1):
        category_counter[category] += 1
        base_price = categories[category]
        price = np.clip(rng.lognormal(np.log(base_price), 0.45), 4.5, 420.0)
        rows.append(
            {
                "product_id": f"P{product_number:04d}",
                "product_name": f"{category} Item {category_counter[category]:02d}",
                "category": category,
                "price": round(float(price), 2),
            }
        )
    return pd.DataFrame(rows)


def generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    """Create fictional customers and assign a hidden behavior archetype.

    The private ``_archetype`` column controls data generation only. It is
    removed before export, so the downstream analysis must infer behavior from
    transactions instead of receiving a pre-made segment label.
    """
    # Archetypes create meaningful differences in frequency, value, and churn.
    archetypes = rng.choice(
        ["high_value_repeat", "loyal", "occasional", "dormant", "recent"],
        size=N_CUSTOMERS,
        p=[0.12, 0.27, 0.31, 0.18, 0.12],
    )
    countries = rng.choice(
        ["United States", "United Kingdom", "Germany", "France", "Canada", "Spain"],
        size=N_CUSTOMERS,
        p=[0.34, 0.18, 0.14, 0.12, 0.12, 0.10],
    )
    channels = rng.choice(
        ["Organic Search", "Paid Search", "Social", "Email", "Referral"],
        size=N_CUSTOMERS,
        p=[0.29, 0.24, 0.19, 0.16, 0.12],
    )
    signup_dates = []
    for archetype in archetypes:
        # The recent cohort signs up later; other cohorts span most of history.
        if archetype == "recent":
            low, high = pd.Timestamp("2025-06-01"), pd.Timestamp("2025-11-20")
        else:
            low, high = START_DATE, pd.Timestamp("2025-06-30")
        signup_dates.append(low + pd.Timedelta(days=int(rng.integers(0, (high - low).days + 1))))
    return pd.DataFrame(
        {
            "customer_id": [f"C{number:04d}" for number in range(1, N_CUSTOMERS + 1)],
            "signup_date": signup_dates,
            "country": countries,
            "acquisition_channel": channels,
            "_archetype": archetypes,
        }
    )


def generate_orders_and_items(
    rng: np.random.Generator, customers: pd.DataFrame, products: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate orders and line items from each customer's behavior profile.

    Purchase gaps, churn probability, basket size, and spend multipliers vary
    by archetype. Customers also favor two categories, which creates a usable
    but imperfect recommendation signal.
    """
    # These profiles shape behavior without making later ML targets deterministic.
    behavior = {
        "high_value_repeat": {"first_delay": 8, "gap": 30, "churn": 0.018, "basket": 3.0, "value": 1.25},
        "loyal": {"first_delay": 14, "gap": 53, "churn": 0.035, "basket": 2.4, "value": 1.05},
        "occasional": {"first_delay": 28, "gap": 105, "churn": 0.090, "basket": 1.8, "value": 0.95},
        "dormant": {"first_delay": 15, "gap": 62, "churn": 0.055, "basket": 2.1, "value": 1.00},
        "recent": {"first_delay": 16, "gap": 58, "churn": 0.060, "basket": 2.0, "value": 1.00},
    }
    category_names = products["category"].unique().tolist()
    product_ids_by_category = {
        category: products.loc[products["category"].eq(category), "product_id"].to_numpy()
        for category in category_names
    }
    product_lookup = products.set_index("product_id")

    order_rows: list[dict] = []
    item_rows: list[dict] = []
    order_number = 1

    for customer_id, signup_date, _country, _channel, archetype in customers.itertuples(index=False, name=None):
        profile = behavior[archetype]
        # Category preference is persistent, with room for exploration.
        favorite_categories = rng.choice(category_names, size=2, replace=False)
        current_date = pd.Timestamp(signup_date) + pd.Timedelta(
            days=max(1, int(rng.gamma(1.8, profile["first_delay"] / 1.8)))
        )
        stop_date = END_DATE
        if archetype == "dormant":
            # Dormant customers stop buying before the dataset ends.
            stop_date = pd.Timestamp("2024-08-01") + pd.Timedelta(days=int(rng.integers(0, 330)))

        while current_date <= min(stop_date, END_DATE):
            order_id = f"O{order_number:06d}"
            item_count = int(np.clip(rng.poisson(profile["basket"] - 1) + 1, 1, 5))
            chosen_products: list[str] = []
            for _ in range(item_count):
                # Most items come from preferred categories; the rest add variety.
                if rng.random() < 0.72:
                    category = rng.choice(favorite_categories, p=[0.72, 0.28])
                else:
                    category = rng.choice(category_names)
                candidates = product_ids_by_category[category]
                popularity_weights = np.linspace(1.7, 0.5, len(candidates))
                popularity_weights /= popularity_weights.sum()
                product_id = str(rng.choice(candidates, p=popularity_weights))
                if product_id in chosen_products and len(candidates) > 1:
                    product_id = str(rng.choice(candidates[candidates != product_id]))
                chosen_products.append(product_id)

            total_amount = 0.0
            for product_id in chosen_products:
                quantity = int(rng.choice([1, 2, 3], p=[0.82, 0.15, 0.03]))
                list_price = float(product_lookup.loc[product_id, "price"])
                unit_price = max(3.0, list_price * profile["value"] * rng.uniform(0.90, 1.06))
                unit_price = round(unit_price, 2)
                total_amount += quantity * unit_price
                item_rows.append(
                    {
                        "order_id": order_id,
                        "product_id": product_id,
                        "quantity": quantity,
                        "unit_price": unit_price,
                    }
                )
            order_rows.append(
                {
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "order_date": current_date.normalize(),
                    "total_amount": round(total_amount, 2),
                }
            )
            order_number += 1

            if rng.random() < profile["churn"]:
                break
            # Gamma-distributed gaps vary naturally around each profile's cadence.
            mean_gap = profile["gap"] * rng.lognormal(0, 0.16)
            gap_days = max(6, int(rng.gamma(2.4, mean_gap / 2.4)))
            current_date += pd.Timedelta(days=gap_days)

    return pd.DataFrame(order_rows), pd.DataFrame(item_rows)


def main() -> None:
    """Generate and save all four public synthetic tables."""
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    products = generate_products(rng)
    customers_internal = generate_customers(rng)
    orders, order_items = generate_orders_and_items(rng, customers_internal, products)
    # Never expose the hidden archetype as a modeling shortcut or target leak.
    customers = customers_internal.drop(columns="_archetype")

    customers.to_csv(OUTPUT_DIR / "customers.csv", index=False, date_format="%Y-%m-%d")
    products.to_csv(OUTPUT_DIR / "products.csv", index=False)
    orders.to_csv(OUTPUT_DIR / "orders.csv", index=False, date_format="%Y-%m-%d")
    order_items.to_csv(OUTPUT_DIR / "order_items.csv", index=False)

    print("Synthetic demo data generated (all records are fictional).")
    print(f"Customers: {len(customers):,}")
    print(f"Products: {len(products):,}")
    print(f"Orders: {len(orders):,}")
    print(f"Order items: {len(order_items):,}")
    print(f"Date range: {orders['order_date'].min().date()} to {orders['order_date'].max().date()}")


if __name__ == "__main__":
    main()


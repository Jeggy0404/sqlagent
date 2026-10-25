"""Builds a small deterministic demo database (customers, products, orders)."""
from __future__ import annotations

import random
import sqlite3

SCHEMA = """
CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id),
                    product_id INTEGER REFERENCES products(id), quantity INTEGER, ordered_on TEXT);
"""


def build(path: str = ":memory:", seed: int = 7) -> sqlite3.Connection:
    rnd = random.Random(seed)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executescript(SCHEMA)
    countries = ["CZ", "DE", "PL", "AT", "SK"]
    conn.executemany("INSERT INTO customers VALUES(?,?,?)",
                     [(i, f"Customer {i}", rnd.choice(countries)) for i in range(1, 41)])
    cats = {"books": (8, 30), "toys": (10, 50), "tools": (15, 120)}
    prods = []
    for i in range(1, 16):
        cat = list(cats)[(i - 1) % 3]
        lo, hi = cats[cat]
        prods.append((i, f"Product {i}", cat, round(rnd.uniform(lo, hi), 2)))
    conn.executemany("INSERT INTO products VALUES(?,?,?,?)", prods)
    orders = []
    for i in range(1, 301):
        m, d = rnd.randint(1, 12), rnd.randint(1, 28)
        orders.append((i, rnd.randint(1, 40), rnd.randint(1, 15), rnd.randint(1, 5), f"2025-{m:02d}-{d:02d}"))
    conn.executemany("INSERT INTO orders VALUES(?,?,?,?,?)", orders)
    conn.commit()
    return conn

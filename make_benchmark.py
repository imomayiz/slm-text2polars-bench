"""
Generate a synthetic benchmark with (text-to-Polars) questions.
Usage:
    python make_benchmark.py
    python make_benchmark.py --out-path path/to/bench.json
"""

import json
from pathlib import Path

import polars as pl


# --- Fixture data (small, realistic)
employees = pl.DataFrame({
    "id": [1, 2, 3, 4, 5, 6, 7, 8],
    "name": ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Heidi"],
    "department": ["Eng", "Eng", "Sales", "Sales", "Eng", "Ops", "Ops", "Sales"],
    "salary": [120000.0, 95000.0, 85000.0, 110000.0, 140000.0, 75000.0, 80000.0, 92000.0],
    "hire_year": [2019, 2021, 2020, 2018, 2022, 2023, 2020, 2021],
})

sales = pl.DataFrame({
    "order_id": list(range(1, 11)),
    "product": ["A", "B", "A", "C", "B", "A", "C", "C", "B", "A"],
    "region": ["NA", "EU", "EU", "NA", "APAC", "NA", "EU", "APAC", "NA", "APAC"],
    "quantity": [10, 5, 7, 3, 12, 8, 4, 6, 9, 11],
    "price": [9.99, 19.99, 9.99, 29.99, 19.99, 9.99, 29.99, 29.99, 19.99, 9.99],
})

customers = pl.DataFrame({
    "customer_id": [1, 2, 3, 4, 5, 6],
    "name": ["Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne"],
    "country": ["US", "US", "UK", "DE", "US", "US"],
})

orders = pl.DataFrame({
    "order_id": [101, 102, 103, 104, 105, 106, 107],
    "customer_id": [1, 1, 2, 3, 4, 5, 2],
    "amount": [500.0, 1500.0, 200.0, 800.0, 2200.0, 1100.0, 300.0],
})

products = pl.DataFrame({
    "product": ["A", "B", "C", "D"],
    "category": ["Hardware", "Software", "Hardware", "Services"],
    "cost": [4.00, 10.00, 15.00, 20.00],
})


def schema_of(df: pl.DataFrame) -> dict[str, str]:
    return {c: str(dt) for c, dt in df.schema.items()}


def data_of(df: pl.DataFrame) -> list[dict]:
    return df.to_dicts()


def serialize_expected(result) -> dict:
    if isinstance(result, pl.DataFrame):
        return {
            "kind": "dataframe",
            "value": result.to_dicts(),
            "columns": result.columns,
        }
    if isinstance(result, pl.Series):
        return {"kind": "series", "value": result.to_list(), "name": result.name}
    return {"kind": "scalar", "value": result}


def build_item(qid, question, category, difficulty,
               frames: dict[str, pl.DataFrame],
               reference_code: str,
               order_matters: bool = False):
    # Execute reference to compute gold
    ns = {"pl": pl, **frames}
    exec(reference_code, ns)
    result = ns["result"]
    return {
        "id": qid,
        "question": question,
        "category": category,
        "difficulty": difficulty,
        "schema": {name: schema_of(df) for name, df in frames.items()},
        "data": {name: data_of(df) for name, df in frames.items()},
        "reference_code": reference_code,
        "expected_output": serialize_expected(result),
        "tolerance": {"order_matters": order_matters, "float_atol": 1e-6},
    }


items = [
    # --- Simple filters ---
    build_item(
        "q001",
        "How many employees are in the Engineering department?",
        "filter", "easy",
        {"df": employees},
        'result = df.filter(pl.col("department") == "Eng").height',
    ),
    build_item(
        "q002",
        "List the names of employees with salary above 100000.",
        "filter", "easy",
        {"df": employees},
        'result = df.filter(pl.col("salary") > 100000).select("name")',
    ),

    # --- Aggregation ---
    build_item(
        "q003",
        "What is the average salary across all employees?",
        "agg", "easy",
        {"df": employees},
        'result = df.select(pl.col("salary").mean()).item()',
    ),
    build_item(
        "q004",
        "What is the maximum salary in the Sales department?",
        "agg", "easy",
        {"df": employees},
        'result = df.filter(pl.col("department") == "Sales").select(pl.col("salary").max()).item()',
    ),

    # --- Group by + agg ---
    build_item(
        "q005",
        "Average salary per department.",
        "groupby", "medium",
        {"df": employees},
        'result = df.group_by("department").agg(pl.col("salary").mean().alias("avg_salary")).sort("department")',
    ),
    build_item(
        "q006",
        "Total revenue per region, where revenue = quantity * price.",
        "groupby", "medium",
        {"df": sales},
        (
            'result = (df.with_columns((pl.col("quantity") * pl.col("price")).alias("revenue"))'
            '.group_by("region").agg(pl.col("revenue").sum()).sort("region"))'
        ),
    ),

    # --- Sort + limit ---
    build_item(
        "q007",
        "Top 3 highest-paid employees. Return name and salary, sorted by salary descending.",
        "sort_limit", "medium",
        {"df": employees},
        'result = df.select(["name", "salary"]).sort("salary", descending=True).head(3)',
        order_matters=True,
    ),

    # --- Join ---
    build_item(
        "q008",
        "Total order amount per country.",
        "join", "medium",
        {"orders": orders, "customers": customers},
        (
            'result = (orders.join(customers, on="customer_id")'
            '.group_by("country").agg(pl.col("amount").sum().alias("total"))'
            '.sort("country"))'
        ),
    ),
    build_item(
        "q009",
        "Name of the customer with the largest single order.",
        "join", "hard",
        {"orders": orders, "customers": customers},
        (
            'result = (orders.join(customers, on="customer_id")'
            '.sort("amount", descending=True).select("name").head(1).item())'
        ),
    ),

    # --- Multi-step ---
    build_item(
        "q010",
        "Number of distinct products sold in the EU region.",
        "filter_agg", "medium",
        {"df": sales},
        'result = df.filter(pl.col("region") == "EU").select(pl.col("product").n_unique()).item()',
    ),

    # --- Simple select / projection ---
    build_item(
        "q011",
        "Return only the name and department columns for all employees, sorted by name.",
        "select", "easy",
        {"df": employees},
        'result = df.select(["name", "department"]).sort("name")',
        order_matters=True,
    ),

    # --- Multi-condition filter ---
    build_item(
        "q012",
        "List the names of employees hired in 2020 or 2021 who work in Engineering or Sales, sorted alphabetically.",
        "filter", "easy",
        {"df": employees},
        (
            'result = (df.filter(pl.col("hire_year").is_in([2020, 2021]) '
            '& pl.col("department").is_in(["Eng", "Sales"]))'
            '.select("name").sort("name"))'
        ),
        order_matters=True,
    ),

    # --- String operations ---
    build_item(
        "q013",
        "Return the names of employees whose name starts with the letter E or F, sorted alphabetically.",
        "string", "medium",
        {"df": employees},
        (
            'result = (df.filter(pl.col("name").str.starts_with("E") '
            '| pl.col("name").str.starts_with("F"))'
            '.select("name").sort("name"))'
        ),
        order_matters=True,
    ),

    # --- Group by with multiple aggregations ---
    build_item(
        "q014",
        "For each department, return the headcount, min salary, max salary, and mean salary, sorted by department.",
        "groupby", "medium",
        {"df": employees},
        (
            'result = (df.group_by("department")'
            '.agg([pl.len().alias("headcount"),'
            ' pl.col("salary").min().alias("min_salary"),'
            ' pl.col("salary").max().alias("max_salary"),'
            ' pl.col("salary").mean().alias("avg_salary")])'
            '.sort("department"))'
        ),
    ),

    # --- Group by + having-style filter ---
    build_item(
        "q015",
        "Which departments have more than 2 employees? Return department and headcount, sorted by department.",
        "groupby", "medium",
        {"df": employees},
        (
            'result = (df.group_by("department").agg(pl.len().alias("headcount"))'
            '.filter(pl.col("headcount") > 2).sort("department"))'
        ),
    ),

    # --- Window function ---
    build_item(
        "q016",
        "Rank employees by salary within each department (rank 1 = highest salary). Return name, department, salary, and rank, sorted by department then rank.",
        "window", "hard",
        {"df": employees},
        (
            'result = (df.with_columns('
            'pl.col("salary").rank(method="ordinal", descending=True)'
            '.over("department").cast(pl.Int64).alias("rank"))'
            '.select(["name", "department", "salary", "rank"])'
            '.sort(["department", "rank"]))'
        ),
        order_matters=True,
    ),

    # --- Anti-join ---
    build_item(
        "q017",
        "Return the names of customers who have not placed any orders, sorted alphabetically.",
        "join", "medium",
        {"orders": orders, "customers": customers},
        (
            'result = (customers.join(orders, on="customer_id", how="anti")'
            '.select("name").sort("name"))'
        ),
        order_matters=True,
    ),

    # --- Join with additional fixture ---
    build_item(
        "q018",
        "Total quantity sold per product category, sorted by category.",
        "join", "medium",
        {"sales": sales, "products": products},
        (
            'result = (sales.join(products, on="product")'
            '.group_by("category").agg(pl.col("quantity").sum().alias("total_quantity"))'
            '.sort("category"))'
        ),
    ),

    # --- Percent of total ---
    build_item(
        "q019",
        "For each region, compute its revenue (quantity * price) as a percentage of total revenue. Return region and pct, sorted by pct descending.",
        "window", "hard",
        {"df": sales},
        (
            'result = (df.with_columns((pl.col("quantity") * pl.col("price")).alias("rev"))'
            '.group_by("region").agg(pl.col("rev").sum())'
            '.with_columns((pl.col("rev") / pl.col("rev").sum() * 100).alias("pct"))'
            '.select(["region", "pct"]).sort("pct", descending=True))'
        ),
        order_matters=True,
    ),

    # --- Multi-column sort ---
    build_item(
        "q020",
        "Return name, department, and salary for all employees, sorted by department ascending and salary descending.",
        "sort", "easy",
        {"df": employees},
        (
            'result = (df.select(["name", "department", "salary"])'
            '.sort(["department", "salary"], descending=[False, True]))'
        ),
        order_matters=True,
    ),

        # --- Conditional column (when/then/otherwise) ---
    build_item(
        "q021",
        "Classify employees as 'high' if salary >= 100000, otherwise 'low'. Return name and class, sorted by name.",
        "expression", "easy",
        {"df": employees},
        (
            'result = (df.with_columns('
            'pl.when(pl.col("salary") >= 100000).then(pl.lit("high")).otherwise(pl.lit("low")).alias("class"))'
            '.select(["name", "class"]).sort("name"))'
        ),
        order_matters=True,
    ),

    # --- Multiple aggregations with filter inside agg ---
    build_item(
        "q022",
        "For each department, compute average salary of employees hired after 2020.",
        "groupby", "medium",
        {"df": employees},
        (
            'result = (df.filter(pl.col("hire_year") > 2020)'
            '.group_by("department")'
            '.agg(pl.col("salary").mean().alias("avg_salary"))'
            '.sort("department"))'
        ),
    ),

    # --- Join + filter on joined column ---
    build_item(
        "q023",
        "Total order amount for US customers only.",
        "join", "medium",
        {"orders": orders, "customers": customers},
        (
            'result = (orders.join(customers, on="customer_id")'
            '.filter(pl.col("country") == "US")'
            '.select(pl.col("amount").sum()).item())'
        ),
    ),

    # --- Distinct + count ---
    build_item(
        "q024",
        "How many unique regions are present in the sales data?",
        "agg", "easy",
        {"df": sales},
        'result = df.select(pl.col("region").n_unique()).item()',
    ),

    # --- Computed column reused ---
    build_item(
        "q025",
        "Compute revenue (quantity * price) and return the average revenue per order.",
        "agg", "medium",
        {"df": sales},
        (
            'result = (df.with_columns((pl.col("quantity") * pl.col("price")).alias("rev"))'
            '.select(pl.col("rev").mean()).item())'
        ),
    ),

    # --- Window with cumulative sum ---
    build_item(
        "q026",
        "For each region, compute cumulative revenue ordered by order_id. Return order_id, region, and cumulative revenue.",
        "window", "hard",
        {"df": sales},
        (
            'result = (df.with_columns((pl.col("quantity") * pl.col("price")).alias("rev"))'
            '.sort("order_id")'
            '.with_columns(pl.col("rev").cum_sum().over("region").alias("cum_rev"))'
            '.select(["order_id", "region", "cum_rev"]))'
        ),
        order_matters=True,
    ),

    # --- Semi join ---
    build_item(
        "q027",
        "Return names of customers who have placed at least one order, sorted alphabetically.",
        "join", "medium",
        {"orders": orders, "customers": customers},
        (
            'result = (customers.join(orders, on="customer_id", how="semi")'
            '.select("name").sort("name"))'
        ),
        order_matters=True,
    ),

    # --- Top-k within group ---
    build_item(
        "q028",
        "For each department, return the highest paid employee (name and salary).",
        "groupby", "hard",
        {"df": employees},
        (
            'result = (df.sort("salary", descending=True)'
            '.group_by("department")'
            '.agg([pl.col("name").first(), pl.col("salary").first()])'
            '.sort("department"))'
        ),
    ),

    # --- Ratio between groups ---
    build_item(
        "q029",
        "What is the ratio of total salary of Engineering to total salary of Sales?",
        "agg", "hard",
        {"df": employees},
        (
            'eng = df.filter(pl.col("department") == "Eng").select(pl.col("salary").sum()).item()\n'
            'sales_sum = df.filter(pl.col("department") == "Sales").select(pl.col("salary").sum()).item()\n'
            'result = eng / sales_sum'
        ),
    ),

    # --- Filter using aggregation (subquery-style) ---
    build_item(
        "q030",
        "Return employees whose salary is above the overall average salary. Return name and salary sorted descending.",
        "filter", "hard",
        {"df": employees},
        (
            'avg_salary = df.select(pl.col("salary").mean()).item()\n'
            'result = (df.filter(pl.col("salary") > avg_salary)'
            '.select(["name", "salary"]).sort("salary", descending=True))'
        ),
        order_matters=True,
    ),
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic Polars benchmark.")
    parser.add_argument("--out-path", type=str, default=str(Path(__file__).parent / "bench.json"),
                        help="Output path for benchmark JSON.")
    args = parser.parse_args()
    out_path = Path(args.out_path)
    with open(out_path, "w") as f:
        json.dump(items, f, indent=2, default=str)
    print(f"Wrote {len(items)} items to {out_path}")

    # Sanity-print categories + expected values
    for it in items:
        print(f"  [{it['id']}] ({it['category']:12s} {it['difficulty']:6s}) "
              f"{it['question'][:60]}")

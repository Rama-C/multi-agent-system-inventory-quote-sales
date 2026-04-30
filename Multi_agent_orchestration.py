import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df =pd.read_csv(os.path.join(BASE_DIR,"quote_requests.csv"))
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv(os.path.join(BASE_DIR,"quotes.csv"))
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions t
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_risk_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory at minimul stock level as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with stock at reordering level are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    # Only items with stock less than their reuired minimum level are included in the result.
    query = """
        SELECT
            t.item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock,
            i.min_stock_level
        FROM transactions t 
        JOIN inventory i ON t.item_name = i.item_name
        WHERE t.item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY t.item_name, i.min_stock_level
        HAVING stock <= i.min_stock_level
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return {
        row["item_name"]: {
            "current_stock": row["stock"],
            "min_stock_level": row["min_stock_level"]
        }
        for _, row in result.iterrows()
    }



def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)


from dotenv import load_dotenv
from dataclasses import dataclass, field, asdict
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
   # api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key
)

# Paper Company State Management


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent




@tool 
def check_stock_levels(as_of_date: str) -> str:
    """
    Retreives all the available stocks from the database, as of the provided date

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.
    
    Returns:
        a formatted string text of items and its current stock
    """
    # get the inventory  as of date
    inventory  = get_all_inventory(as_of_date) 
    results = []

    # extract the relevant details like item name and its current stock from the results
    for item, stock in inventory.items():
        results.append(f"{item} : {stock}")
    output = "\n".join(results)
    return output

@tool 
def get_min_stocks(as_of_date: str) -> str:
    """
    Retreives the stocks at reordering stages from the database, as of the provided date

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.
    
    Returns:
         a formatted string text of items and its current stock
    """
    # get the inventory at reordering stage as of date
    inventory  = get_risk_inventory(as_of_date) 
    results = []
    # extract the relevant details like item name, its current stock and the minimum stock level from the results
    for item, data in inventory.items():
        results.append(f"{item} : {data['current_stock']} units (min: {data['min_stock_level']})")
    
    output = "\n".join(results)
    return output

class InventoryAgent(ToolCallingAgent):
    """Agent responsible for checking the inventories and identify what needs restocking
    
    Responsibilities:
    - Checks current stock levels across all items
    - identifies which items have fallen below their minimum reorder threshold
    - provides a financial snapshot of the company's asset position.
    
    Tools: 
    1. check_stock_levels: returns current stock count for every item in inventory as of a given date.
    2. get_min_stocks: returns only items that have fallen below their minimum stock threshold — the reorder watchlist

    
    """
    
    def __init__(self, model):
        super().__init__(
            tools=[check_stock_levels, get_min_stocks],
            model=model,
            name="Inventory_processor",
            #max_steps = 5,
            description="Agent responsible for checking the stocks and restock them when needed."
        )

# Tools for quoting agent
@tool 
def check_stock_for_requests(item_name: str, as_of_date: Union[str, datetime])-> str:
    """
    Checks whether the specific item is available to fulfill it

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.
    Returns:
        a formatted string text of items and its current stock
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # get the stock level of the requested item
    stocks  = get_stock_level(item_name, as_of_date) 
    
    #item = stocks.iloc[0]['item_name']
    current_stock = stocks.iloc[0]['current_stock']
    return f"{item_name} : {current_stock}"

@tool 
def get_the_quoteprice(customer_request: str, limit: int = 5) -> str:
    """
    Retrieves the quoteprice for the search terms provided. The results are limited 
    by the limit parameter. 

    Args:
        customer_request : a text string confirming what the customer wants
        limit (int, optional): Maximum number of quote records to return. Default is 5.
    Returns:
        a formatted string text of items and its current stock
    """
    # customer_request is split into words and the most common words are skipped 
    stop_words = {"i", "need", "for", "an", "a", "the", "of", "to", "and", "in", 
                  "want", "please", "units", "sheets", "rolls", "pieces", "some",
                  "our", "my", "we", "us", "by", "with", "at", "on", "is", "are"}
    words = customer_request.lower().replace(",", " ").replace(".", " ").split()
    search_terms = [word for word in words if word not in stop_words and len(word) > 2]
    
    # check valid search terms are found
    if not search_terms:
        return "No valid search terms found in request."
    
    # pass the search terms to get the matching quotes
    quotes  = search_quote_history(search_terms, limit) 
    if not quotes:
        return "No matching quotes found in history."
    lines = []

    # extract the relevant info like original request. total_amount and the quote explanation
    for quote in quotes:
        lines.append(
            f"Request: {quote['original_request']}\n"
            f"Amount: {quote['total_amount']}\n"
            f"Explanation: {quote['quote_explanation']}"
        )
    output = "\n---\n".join(lines)
    return output

@tool 
def calculate_quote(item_name: str, quantity: int, as_of_date : str) -> str:
    """
    Calculate the quote for the requested items and the quantity. Discounts are applied based on the quantity

    Args:
        item_name (str): name of the item requested.
        quantity (int): The number of items or units to calculate the quote for.
        as_of_date (str): order delivery date
    
    Returns:
        a format string with detailed information of requested item, quantity, unit price, subtotal
        discount if any and then final price

        < 100 units   → 0% discount
        100-499 units → 5% discount
        500+ units    → 10% discount

    """
    # check the stock level for the requested item
    stock_df = get_stock_level(item_name, as_of_date)
    current_stock = stock_df.iloc[0]['current_stock']

    # get unit price for the requested item from the inventory table
    inventory_df = pd.read_sql(
        "SELECT unit_price FROM inventory WHERE item_name LIKE :item_name",
        db_engine,
        params={"item_name": item_name}
    )

    # handle the error if the item is not found
    if inventory_df.empty:  
        return f"Item '{item_name}' not found in inventory."
    
   
    # check enough stock is available 
    if current_stock < quantity:
        return f"Insufficient stock for {item_name}. Available: {current_stock} units."
   

    # retreive the unit price and calculate the subtotal
    unit_price = inventory_df.iloc[0]['unit_price']
    subtotal = unit_price * quantity

    # apply the discount based on the quantity as follows. 
    # apply 0% discount for orders under 100 units; 
    # 5% for orders with 100 -499 units; 
    # 10% for bulk orders over 500 units
    if quantity < 100:
        discount_per = 0
        discount_desc = " "
    elif 100 <= quantity < 500:
        discount_per = 5
        discount_desc = "(order between 100 and 499 units)"
    elif quantity >= 500 :
        discount_per = 10
        discount_desc = "(bulk order over 500 units)"
    else:
        discount_per = 0
        discount_desc = " "

    # apply the discount
    final_price = subtotal - (subtotal * discount_per / 100)
    
    
    # share the relevant details
    return (
            f"Item : {item_name}\n"
            f"Quantity: {quantity}\n"
            f"Unit Price : ${unit_price:.2f}\n"
            f"Subtotal : ${subtotal:.2f}\n"
            f"Discount: {discount_per}% {discount_desc}\n"
            f"Final price: ${final_price:.2f}"
    )

class QuotingAgent(ToolCallingAgent):
    """Agent responsible for understnding the customer request, check the stock is available to fulfill it
    get past quotes for pricing reference, calculate the total price and apply discount if qualify

    Responsibilities:
    - Checks whether requested items are available in sufficient quantity
    - references past quote history for pricing context
    - calculates the final price applying tiered discounts based on order size. 
    - Never writes to the database — read only.
    
    Tools are:
    1.check_stock_for_requests: checks if a specific item has enough stock to fulfill a  request
    2. get_the_quoteprice:  searches past quote history for similar requests to use as a pricing reference
    3. calculate_quote:  calculates the full price for an item including tiered discount based on quantity
    """
    
    def __init__(self, model):
        super().__init__(
            tools=[check_stock_for_requests,get_the_quoteprice, calculate_quote],
            model=model,
            name="quote_generator",
            #max_steps = 5,
            description="Agent responsible for generating the quotes for the requested items"
        )
# Tools for ordering agent

@tool
def check_cash_balance(as_of_date : str) -> str:
    """
    Checks the cash balance as of date to place an order

    Args:
         as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the cash check date.
    
    Returns:
        available cash in a text string
    """
    # check the available cash to proceed with a transaction
    cash = get_cash_balance(as_of_date)
    return (
        f"Cash : ${cash}"
    )
@tool
def process_sale(item_name: str, quantity: int, as_of_date : str) -> str:
    """
    Confirm enough stocks exists and record the sale transaction

    Args:
        item_name (str): name of the item requested
        quantity (int): number of units requested
        as_of_date (str): order delivery date
    Returns:
        a formatted string with the item, quantity and total charged
    """
    # get the stock level of the requested item
    stock_df = get_stock_level(item_name, as_of_date)
    current_stock = stock_df.iloc[0]['current_stock']

    # get unit price for the requested item from inventory table
    inventory_df = pd.read_sql(
        "SELECT unit_price FROM inventory WHERE item_name LIKE :item_name",
        db_engine,
        params={"item_name": item_name}
    )

    # handle the error if the requested item is not found
    if inventory_df.empty:  
        return f"Item '{item_name}' not found in inventory."
    
    # retrieve the unit_price from the results
    unit_price = inventory_df.iloc[0]['unit_price']
    
    # check enough stock is available to proceed with the sale
    if current_stock < quantity:
        return f"There is no enough stock for {item_name} with stock count {current_stock} so a transcation cannot be processed"
    
    # calculate the subtotal for the purchase
    subtotal = unit_price * quantity

    # apply the discount as follows. apply 0% discount for orders under 100 units; 5% for orders with 100 -499 units; 10% for orders over 500 units
    if quantity < 100:
        discount_pct = 0
    elif quantity < 500:
        discount_pct = 5
    else:
        discount_pct = 10

    total = subtotal * (1 - discount_pct / 100)

    # Record the transaction and retrieve the order id
    insert_id = create_transaction(item_name,'sales',quantity, total,as_of_date)

    result = pd.read_sql(
            """
            SELECT rowid as transaction_id FROM transactions 
            WHERE item_name = :item_name 
            AND transaction_type = 'sales'
            ORDER BY rowid DESC 
            LIMIT 1
            """,
            db_engine,
            params={
                "item_name": item_name,
                "transaction_type": 'sales'
            }
    )
    new_id =  int(result.iloc[0]["transaction_id"])
    
    # share the details of the order placed if the sale was succesful, otherwise return with the message
    if insert_id:
        return (
            f"Transaction processed successfully.\n"
            f"Transaction ID: {new_id}\n"
            f"Item: {item_name}\n"
            f"Quantity: {quantity}\n"
            f"Discount: {discount_pct}%\n"
            f"Total charged: ${total:.2f}"
        )
    else:
        return f"Transaction was recorded but no ID was returned."
    
@tool 
def restock_item(item_name: str, quantity: int, as_of_date : str) -> str:
    """ 
    Buy from supplier when stock is low

    Args:
        item_name (str): name of the item requested
        quantity (int): number of units requested
        as_of_date (str): order delivery date
    Returns:
        a formatted string with the cost and expected delivery date

    """
    # get the unit_price for the item_name from the inventory table
    inventory_df = pd.read_sql(
       "SELECT unit_price FROM inventory WHERE item_name LIKE :item_name",
        db_engine,
        params={"item_name": item_name}
    )

    # handle the error if the requested item is not found
    if inventory_df.empty:  
        return f"Item '{item_name}' not found in inventory."
    
    # retrieve the unit_price from the results and calculate the restock_total
    unit_price = inventory_df.iloc[0]['unit_price']
    restock_total = unit_price * quantity

    # check the available cash before restocking
    cash = get_cash_balance(as_of_date)

    # check cash is enough for restocking. if  yes - place order, publish the order id and delivery date, if No- return with a message
    if cash < restock_total:
        return "We don't have enough cash to restock the items, at the moment"
    else:
        insert_id = create_transaction(item_name,'stock_orders',quantity, restock_total,as_of_date)
        result = pd.read_sql(
            """
            SELECT rowid as transaction_id FROM transactions 
            WHERE item_name = :item_name 
            AND transaction_type = 'stock_orders'
            ORDER BY rowid DESC 
            LIMIT 1
            """,
            db_engine,
            params={
                "item_name": item_name,
                "transaction_type": 'stock_orders'
            }
    )
        new_id =  int(result.iloc[0]["transaction_id"])
        delivery_date = get_supplier_delivery_date(as_of_date, quantity)
    
        return f"The Order with the order_id {new_id}  has been placed and the delivery_date is {delivery_date}"

class OrderingAgent(ToolCallingAgent):
    """Agent responsible for checking the stocks, available cash, record the sales
    transaction

    Responsibilities:
    - Verifies cash availability before any purchase
    - records sale transactions to the database
    - triggers restocking when stock drops below minimum levels
    - returns supplier delivery dates. 
    - This is the only agent that writes sales and stock order transactions.

    Tools are:
    1. check_cash_balance: returns available cash to confirm we can afford a restock
    2. process_sale: verifies stock exists, applies discount, records the sale transaction in the database
    3. restock_item: checks cash, places a stock order with the supplier, records the purchase transaction and returns delivery date
    """
    
    def __init__(self, model):
        super().__init__(
            tools=[check_cash_balance,check_stock_levels, process_sale, restock_item],
            model=model,
            name="order_processor",
            #max_steps = 5,
            description="Agent responsible for processing orders, verifying stock and cash, recording sales, and restocking when needed."
        )

# Set up your agents and create an orchestration agent that will manage them.
# ======= Orchestrator =======


class Orchestrator(ToolCallingAgent):
    """ 
    Receives every customer request and delegates to the three specialist agents - Inventory agent, Quoting agent and Ordering agent.

    Orchestrator tools are:
    1. manage_inventory: routes to Inventory Agent — triggers full stock check and reorder assessment
    2. manage_quotes: routes to Quoting Agent — generates price quotes for all items in a customer request
    3. manage_order: routes to Ordering Agent — processes the confirmed sale and handles restocking if needed
    
    """
    
    def __init__(self, model):

        self.model = model
        
        #  Initialize specialist agents before defining tools
        self.inventory_processor = InventoryAgent(model)
        self.quoteprice_processor = QuotingAgent(model)
        self.order_processor = OrderingAgent(model)

        # Create coordination tools that route requests to different agents
        # Wrap each agent as a tool so the orchestrator LLM
        # can invoke them via natural language reasoning
        @tool
        def manage_inventory(as_of_date: str) -> str:
            """Check and manage inventory as of the provided date.
            
            Args:
                 as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.
       
            Returns:
                Inventory management result
            """
            # Route this request to the InventoryAgent
            result = self.inventory_processor.run(
                f"""
            Check the inventory for the date "{as_of_date}"

            1. Use check_stock_levels to get all current stock levels.
            2. Use get_min_stocks to identify items at or below their minimum stock level.
            3. Return a clear summary with two sections:
            - AVAILABLE STOCK: all items and quantities
            - NEEDS RESTOCKING: items below minimum, with current and minimum levels
        
            Be specific — the ordering agent will use NEEDS RESTOCKING to reorder items.
            """
            )
            return result
        @tool
        def manage_quotes(customer_request: str, as_of_date: str) -> str:
            """Receives a customer request, identifies the list of items requested and the delivery date.
            calculate total cost for the customer's requested items and quantity, apply discounts where appropriate.
            use the past quotes as a reference when deciding the total cost or discounts.
            
            Args:
                customer_request (str): a text message detailing customer's requested items.
                as_of_date (str): ISO-formatted date string (YYYY-MM-DD) for the quote date.
            Returns:
                Quote management result
            """
            # Route this request to the QuotingAgent
            return self.quoteprice_processor.run(
                 f"""
                Available inventory items (use ONLY these exact names):
                {check_stock_levels(as_of_date)}

                customer request is {customer_request}

                1. identify the list of items requested — use the EXACT item name 
                as it appears in inventory (e.g. 'A4 paper', 'Kraft paper')
                not the full customer description
                2. number of units requested
                3. expected order fulfillment date

                
                Use get_the_quoteprice, check_stock_for_requests, calculate_quote and the above extracted details 
                and return the result of the price quotes
            
            """)
        
        @tool
        def manage_order(item_name: str, quantity: int, as_of_date : str) -> str:
            """Take a confirmed order, check we can fulfill it and record the transaction.
            When the stock is low, restock them and keep track of supply info
            Args:
                item_name (str): name of the item to order.
                quantity (int): number of units requested.
                as_of_date (str): ISO-formatted date string (YYYY-MM-DD) for the order date.

            Returns:
                order confirmation with transaction id and delivery date
            """
            # check the stock before attempting to fulfill the order
            available_stock = check_stock_levels(as_of_date)

            # Route this request to the OrderingAgent
            return self.order_processor.run(
                 f"""
            Available inventory (use ONLY these exact item names):
            {available_stock}

            Process order for:
            - Item: {item_name}
            - Quantity: {quantity}
            - Date: {as_of_date}
            
            Steps:
            1. Use check_cash_balance to confirm sufficient funds.
            2. Use process_sale to record the sale if stock is sufficient.
            3. If stock is insufficient after the sale, use restock_item to replenish.
            4. Return confirmation with transaction ID and delivery date if restocked.
                        
            """)
        super().__init__(
            tools=[ manage_inventory, manage_quotes, manage_order ],  # Add the coordination tools
            model=model,
            name="orchestrator",
            #max_steps = 15,
            description="""
            You are the orchestrator for a paper mill company.
            You coordinate between the  inventory manager, ordering processor and quote price operator
            
            For each query, follow this workflow:
            1. Use manage_inventory to check the inventory
            2. Use manage_quotes to generate the quote for the items 
            requested and apply discount if qualify
            3. Use manage_order to check the stocks and restock them when needed.
            4. Make sure enough cash is available for a purchase and record the transactions
            5. Keep track of supplier details to track the orders.
           
            
            Always provide clear responses to the queries.
            """,
            
        )


    def process_request(self, request: str) -> str:
        """
        Process a request through coordinated agent workflow.
        Entry point for processing a customer request through the 
        multi-agent workflow.

        Coordinates the following steps:
        1. Inventory check — verifies current stock levels
        2. Quote generation — calculates pricing with discounts
        3. Order processing — records transactions and handles restocking

        Args:
            request (str): The customer's request including items, 
                        quantities and date of request.

        Returns:
            str: A customer-facing response with quote details, 
                transaction IDs, and delivery dates. If the order 
                cannot be fulfilled, returns an explanation of why.
        """
       
        try:
            print("\n--- Processing New Order ---")
            
            # Use the orchestrator's own coordination workflow
            context = f"""
            The request: "{request}"
            
            Process this order by coordinating with our specialized agents:
            1. Use manage_inventory to check current stock levels.
            2. Use manage_quotes to generate a price quote for ALL items in the request,
            applying discounts based on quantity.
            3. For EACH item in the request, call manage_order SEPARATELY —
            one manage_order call per item, do not batch them together.
            4. When restocking is needed, get the supplier delivery date and 
            record the transaction.

            If at any step we cannot fulfill an item, explain why to the customer.
            Return a clear summary showing each item, its transaction ID, 
            discount applied, and total charged.

            """
            
            return self.run(context)
            
        except Exception as e:
            print(f"Error processing order: {str(e)}")
            
            return "I'm sorry, we encountered an error processing your order. Please try again or contact customer service."



# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv(os.path.join(BASE_DIR,"quote_requests_sample.csv"))
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
        quote_requests_sample = quote_requests_sample.reset_index(drop=True) 
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    orchestrator = Orchestrator(model)

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = orchestrator.process_request(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv(os.path.join(BASE_DIR,"Order_summary_output.csv"), index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()

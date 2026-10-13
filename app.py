import os
import ssl
import smtplib
import sqlite3
from datetime import date, timedelta
from email.message import EmailMessage
import pickle
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="MedAlert AI | Smart Inventory",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}
.stApp { background: #F4F7FB; }
[data-testid="stSidebar"] {
    background: #102238;
}
[data-testid="stSidebar"] * { color: #EDF4FC; }
.hero {
    background: linear-gradient(120deg, #102238, #176D73);
    padding: 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 24px;
}
.hero h1 { color: white; margin-bottom: 5px; }
.hero p { color: #D3E9E9; margin-bottom: 0; }
.metric {
    background: white;
    padding: 20px;
    border: 1px solid #E3EAF2;
    border-radius: 15px;
    min-height: 125px;
}
.metric-label { color: #738297; font-size: 13px; }
.metric-value { color: #132E49; font-size: 29px; font-weight: 800; }
.metric-note { color: #728096; font-size: 12px; }
.section {
    color: #17334F;
    font-size: 21px;
    font-weight: 700;
    margin: 16px 0 10px;
}
div.stButton > button {
    background: #147D78;
    color: white;
    border: 0;
    border-radius: 9px;
    font-weight: 600;
    min-height: 42px;
}
div.stButton > button:hover {
    background: #10635F;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------
DB_PATH = "database/medalert.db"
os.makedirs("database", exist_ok=True)
os.makedirs("reports", exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine TEXT NOT NULL,
                category TEXT,
                batch TEXT UNIQUE NOT NULL,
                stock INTEGER NOT NULL,
                min_stock INTEGER NOT NULL,
                expiry TEXT NOT NULL,
                daily_sales REAL NOT NULL,
                unit_price REAL NOT NULL,
                supplier_email TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine TEXT,
                batch TEXT,
                quantity INTEGER,
                status TEXT,
                created_at TEXT
            )
        """)

        count = conn.execute(
            "SELECT COUNT(*) FROM inventory"
        ).fetchone()[0]

        if count == 0:
            sample = [
                ("Paracetamol 500mg", "Pain Relief", "PAR101",
                 120, 50, date.today()+timedelta(days=120), 8, 2.5, ""),
                ("Amoxicillin 250mg", "Antibiotic", "AMX202",
                 25, 40, date.today()+timedelta(days=20), 5, 8.0, ""),
                ("Cetirizine 10mg", "Allergy", "CET303",
                 15, 30, date.today()+timedelta(days=60), 4, 3.0, ""),
                ("Vitamin C 500mg", "Supplement", "VIT404",
                 90, 25, date.today()+timedelta(days=240), 3, 4.0, ""),
                ("Ibuprofen 200mg", "Pain Relief", "IBU505",
                 18, 35, date.today()+timedelta(days=12), 6, 5.0, ""),
                ("Antacid Tablets", "Digestive Health", "ANT606",
                 75, 20, date.today()+timedelta(days=180), 2, 2.0, ""),
                ("Oral Rehydration Salts", "Hydration", "ORS707",
                 42, 30, date.today()+timedelta(days=75), 5, 6.0, ""),
            ]

            conn.executemany("""
                INSERT INTO inventory
                (medicine, category, batch, stock, min_stock,
                 expiry, daily_sales, unit_price, supplier_email)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (n, c, b, s, m, e.isoformat(), d, p, se)
                for n, c, b, s, m, e, d, p, se in sample
            ])

initialize_database()

def load_inventory():
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM inventory", conn)
    if not df.empty:
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        df["days_to_expiry"] = (
            pd.to_datetime(df["expiry"]) - pd.Timestamp(date.today())
        ).dt.days
    return df


def get_reorder_quantity(row):
    try:
        with open("models/demand_model.pkl", "rb") as file:
            model = pickle.load(file)

        today = date.today()
        features = pd.DataFrame([{
            "day": today.day,
            "weekday": today.weekday(),
            "month": today.month,
            "stock_available": int(row["stock"])
        }])

        predicted = max(0, float(model.predict(features)[0]))
        daily_sales = max(0, int(row["daily_sales"]))
        stock = max(0, int(row["stock"]))
        min_stock = max(0, int(row["min_stock"]))

        # Limit forecast to avoid unrealistic recommendations.
        predicted = min(predicted, daily_sales * 1.2)

        # Estimate demand for 7 days and maintain minimum stock.
        target = max(min_stock, int(predicted * 7) + min_stock)
        suggested_order = max(0, target - stock)

        # Cap recommendation to 14 days of recorded sales.
        max_order = daily_sales * 14
        return min(suggested_order, max_order)

    except Exception as e:
        st.warning(f"ML prediction unavailable: {e}")
        daily_sales = max(0, int(row["daily_sales"]))
        stock = max(0, int(row["stock"]))
        min_stock = max(0, int(row["min_stock"]))

        target = max(min_stock, daily_sales * 7 + min_stock)
        return min(max(0, target - stock), daily_sales * 14)


def create_order_draft(row, quantity):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO orders
            (medicine, batch, quantity, status, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            row["medicine"],
            row["batch"],
            int(quantity),
            "Awaiting approval",
            date.today().isoformat()
        ))

def load_orders():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM orders ORDER BY id DESC", conn
        )

def send_email(subject, body):
    # Configure these values in Streamlit secrets or environment variables.
    sender = st.secrets.get(
        "EMAIL_ADDRESS", os.getenv("EMAIL_ADDRESS", "")
    )
    password = st.secrets.get(
        "EMAIL_APP_PASSWORD", os.getenv("EMAIL_APP_PASSWORD", "")
    )
    recipient = st.secrets.get(
        "ALERT_RECIPIENT", os.getenv("ALERT_RECIPIENT", "")
    )

    if not sender or not password or not recipient:
        return False, (
            "Email settings are missing. Configure EMAIL_ADDRESS, "
            "EMAIL_APP_PASSWORD and ALERT_RECIPIENT first."
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(body)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.login(sender, password)
            server.send_message(msg)
        return True, "Email notification sent successfully."
    except Exception as exc:
        return False, f"Email failed: {exc}"

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 💊 MedAlert AI")
    st.caption("INVENTORY INTELLIGENCE")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Dashboard",
            "Medicine Inventory",
            "Expiry Monitoring",
            "Reorder Center",
            "Add Medicine",
            "Order History",
        ],
    )

    st.divider()
    st.caption("SYSTEM STATUS")
    st.success("Application running")
    st.caption("Inventory stored in SQLite")

# ---------------- CURRENT DATA ----------------
inventory = load_inventory()

if not inventory.empty:
    inventory["reorder_qty"] = inventory.apply(
        get_reorder_quantity, axis=1
    )

low_stock = inventory[
    inventory["stock"] <= inventory["min_stock"]
]
near_expiry = inventory[
    (inventory["days_to_expiry"] >= 0)
    & (inventory["days_to_expiry"] <= 30)
]
expired = inventory[inventory["days_to_expiry"] < 0]

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>💊 MedAlert AI</h1>
        <p>Intelligent medicine inventory, expiry monitoring
        and automated replenishment assistance.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("Medicine Records", len(inventory), "Tracked entries"),
        ("Available Units", f"{inventory['stock'].sum():,}", "Units in stock"),
        ("Low Stock", len(low_stock), "Require review"),
        ("Near Expiry", len(near_expiry), "Within 30 days"),
    ]

    for col, (label, value, note) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown(
            '<div class="section">Stock Levels</div>',
            unsafe_allow_html=True
        )
        fig = px.bar(
            inventory.sort_values("stock"),
            x="stock",
            y="medicine",
            orientation="h",
            color="stock",
            color_continuous_scale=["#A7DCD7", "#147D78"],
            labels={"stock": "Units", "medicine": ""}
        )
        fig.update_layout(
            height=360,
            plot_bgcolor="white",
            paper_bgcolor="white",
            coloraxis_showscale=False,
            margin=dict(l=5, r=5, t=10, b=5)
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(
            '<div class="section">Inventory Categories</div>',
            unsafe_allow_html=True
        )
        category = inventory.groupby(
            "category", as_index=False
        )["stock"].sum()

        fig = px.pie(
            category,
            names="category",
            values="stock",
            hole=0.60,
            color_discrete_sequence=[
                "#147D78", "#4F86C6", "#E9A23B",
                "#8067B7", "#56A6A1", "#D97878"
            ]
        )
        fig.update_layout(
            height=360,
            paper_bgcolor="white",
            margin=dict(l=5, r=5, t=10, b=5)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section">Priority Alerts</div>',
        unsafe_allow_html=True
    )

    if not expired.empty:
        st.error(
            f"{len(expired)} records have passed their expiry date. "
            "Review and remove affected stock from sale."
        )

    if not low_stock.empty:
        for _, row in low_stock.iterrows():
            st.warning(
                f"{row['medicine']}: {row['stock']} units remaining; "
                f"minimum level is {row['min_stock']}."
            )

    if not near_expiry.empty:
        for _, row in near_expiry.iterrows():
            st.info(
                f"{row['medicine']} — batch {row['batch']} expires "
                f"in {row['days_to_expiry']} days."
            )

    st.caption(
        "Initial version uses demonstration data and rule-based reorder "
        "calculations. A trained demand forecasting model will be added "
        "separately."
    )

# ---------------- INVENTORY ----------------
elif page == "Medicine Inventory":
    st.markdown(
        '<div class="hero"><h1>Medicine Inventory</h1>'
        '<p>Search, review and export stock records.</p></div>',
        unsafe_allow_html=True
    )

    query = st.text_input("Search by medicine or batch")
    categories = ["All"] + sorted(inventory["category"].unique().tolist())
    selected = st.selectbox("Category", categories)

    filtered = inventory.copy()

    if query:
        filtered = filtered[
            filtered["medicine"].str.contains(
                query, case=False, na=False
            )
            | filtered["batch"].str.contains(
                query, case=False, na=False
            )
        ]

    if selected != "All":
        filtered = filtered[filtered["category"] == selected]

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.download_button(
        "Download inventory CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "medalert_inventory.csv",
        "text/csv",
    )

# ---------------- EXPIRY ----------------
elif page == "Expiry Monitoring":
    st.markdown(
        '<div class="hero"><h1>Expiry Monitoring</h1>'
        '<p>Identify batches that require review.</p></div>',
        unsafe_allow_html=True
    )

    days = st.selectbox("Expiry window", [30, 60, 90, 180])
    results = inventory[
        (inventory["days_to_expiry"] >= 0)
        & (inventory["days_to_expiry"] <= days)
    ]

    st.metric("Batches to Review", len(results))

    if not expired.empty:
        st.error(f"{len(expired)} records have expired.")

    st.dataframe(
        results[
            ["medicine", "batch", "stock", "expiry", "days_to_expiry"]
        ],
        use_container_width=True,
        hide_index=True,
    )

# ---------------- REORDER CENTER ----------------
elif page == "Reorder Center":
    st.markdown(
        '<div class="hero"><h1>Reorder Center</h1>'
        '<p>Generate replenishment drafts and notify your inventory team.</p></div>',
        unsafe_allow_html=True
    )

    st.info(
        "Low-stock items are detected automatically. Purchase orders "
        "remain drafts until an authorized person approves them."
    )

    if low_stock.empty:
        st.success("No low-stock items currently require review.")
    else:
        reorder = low_stock.copy()
        reorder["Suggested Order"] = reorder.apply(
            get_reorder_quantity, axis=1
        )
        reorder["Estimated Cost"] = (
            reorder["Suggested Order"] * reorder["unit_price"]
        ).round(2)

        st.dataframe(
            reorder[
                [
                    "medicine", "batch", "stock", "min_stock",
                    "daily_sales", "Suggested Order", "Estimated Cost"
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        # Automatically prepare downloadable order recommendations.
        csv = reorder[
            [
                "medicine", "batch", "stock", "Suggested Order",
                "unit_price", "Estimated Cost"
            ]
        ].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download purchase-order draft",
            csv,
            "medalert_purchase_order_draft.csv",
            "text/csv",
        )

        if st.button("Create reorder drafts", use_container_width=True):
            existing = load_orders()
            created = 0

            for _, row in reorder.iterrows():
                already_pending = (
                    not existing.empty
                    and (
                        (existing["batch"] == row["batch"])
                        & (existing["status"] == "Awaiting approval")
                    ).any()
                )

                if not already_pending:
                    create_order_draft(row, row["Suggested Order"])
                    created += 1

            st.success(f"Created {created} new purchase-order draft(s).")
            st.rerun()

        st.divider()
        st.subheader("Email Notification")

        recipient = st.text_input(
            "Notification recipient",
            placeholder="inventory.manager@example.com",
            help="The configured ALERT_RECIPIENT receives the email.",
        )

        if st.button("Send low-stock email alert"):
            # Validate that an email address was supplied in settings.
            configured_recipient = st.secrets.get(
                "ALERT_RECIPIENT",
                os.getenv("ALERT_RECIPIENT", "")
            )

            if not configured_recipient:
                st.error(
                    "Configure ALERT_RECIPIENT in .streamlit/secrets.toml "
                    "or as an environment variable before sending."
                )
            elif recipient and recipient.strip() != configured_recipient.strip():
                st.error(
                    "The entered recipient differs from ALERT_RECIPIENT. "
                    "Update your configuration first."
                )
            else:
                lines = [
                    f"- {r['medicine']} (batch {r['batch']}): "
                    f"{r['stock']} in stock; suggested order "
                    f"{get_reorder_quantity(r)} units"
                    for _, r in reorder.iterrows()
                ]

                success, message = send_email(
                    "MedAlert AI: Low Stock Alert",
                    "The following items need inventory review:\n\n"
                    + "\n".join(lines)
                    + "\n\nPlease review and approve any orders before "
                    "placing them with suppliers."
                )

                if success:
                    st.success(message)
                else:
                    st.error(message)

# ---------------- ADD MEDICINE ----------------
elif page == "Add Medicine":
    st.markdown(
        '<div class="hero"><h1>Add Medicine</h1>'
        '<p>Register a new batch in the inventory database.</p></div>',
        unsafe_allow_html=True
    )

    with st.form("add_medicine"):
        name = st.text_input("Medicine name")
        category = st.selectbox(
            "Category",
            [
                "Pain Relief", "Antibiotic", "Allergy",
                "Supplement", "Digestive Health", "Hydration", "Other"
            ]
        )
        batch = st.text_input("Unique batch number")
        stock = st.number_input("Current stock", min_value=0, step=1)
        minimum = st.number_input("Minimum stock", min_value=0, step=1)
        expiry = st.date_input(
            "Expiry date",
            value=date.today() + timedelta(days=180),
            min_value=date.today()
        )
        sales = st.number_input("Average daily sales", min_value=0.0)
        price = st.number_input("Unit price (₹)", min_value=0.0)
        supplier_email = st.text_input("Supplier email (optional)")

        submitted = st.form_submit_button("Save medicine")

        if submitted:
            if not name.strip() or not batch.strip():
                st.error("Medicine name and batch number are required.")
            else:
                try:
                    with get_connection() as conn:
                        conn.execute("""
                            INSERT INTO inventory
                            (medicine, category, batch, stock, min_stock,
                             expiry, daily_sales, unit_price, supplier_email)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            name.strip(), category, batch.strip(),
                            int(stock), int(minimum), expiry.isoformat(),
                            float(sales), float(price), supplier_email.strip()
                        ))

                    st.success("Medicine saved successfully.")
                    st.rerun()

                except sqlite3.IntegrityError:
                    st.error("This batch number already exists.")

# ---------------- ORDER HISTORY ----------------
elif page == "Order History":
    st.markdown(
        '<div class="hero"><h1>Order History</h1>'
        '<p>Review purchase-order drafts and their status.</p></div>',
        unsafe_allow_html=True
    )

    orders = load_orders()

    if orders.empty:
        st.info("No order drafts have been created yet.")
    else:
        st.dataframe(orders, use_container_width=True, hide_index=True)

        pending_ids = orders.loc[
            orders["status"] == "Awaiting approval", "id"
        ].tolist()

        if pending_ids:
            selected_id = st.selectbox(
                "Select draft to approve for external ordering",
                pending_ids
            )

            if st.button("Mark draft as approved"):
                with get_connection() as conn:
                    conn.execute(
                        "UPDATE orders SET status = ? WHERE id = ?",
                        ("Approved for ordering", int(selected_id))
                    )

                st.success(
                    "Draft marked approved. No supplier order was sent."
                )
                st.rerun()
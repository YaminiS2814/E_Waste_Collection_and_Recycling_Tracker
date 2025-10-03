# app.py
"""
E-Waste Management System - Full Streamlit App
Requires:
    pip install streamlit pandas sqlalchemy psycopg2-binary plotly python-dotenv
Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime
import io
import os


# ensure refresh flag exists
if "refresh_tables" not in st.session_state:
    st.session_state["refresh_tables"] = False

# --------- Configuration (edit if your DB creds differ) ----------
DB_USER = os.getenv("DB_USER", "ewaste_user")
DB_PASS = os.getenv("DB_PASS", "ewaste_pass")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ewaste_db")

CONN_STR = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONN_STR, pool_pre_ping=True)

# --------- Streamlit page config ----------
st.set_page_config(page_title="♻️ E-Waste Management", layout="wide", initial_sidebar_state="collapsed")
# Inject CSS for a modern look
st.markdown(
    """
    <style>
    /* page background + cards */
    .stApp { background: linear-gradient(180deg,#f7fbff 0%, #ffffff 60%); }
    .card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(64,77,106,0.08);
        margin-bottom: 18px;
    }
    .muted { color: #6b7280; }
    .big { font-size:20px; font-weight:600; }
    .metric-row { display:flex; gap:16px; flex-wrap:wrap; }
    .pill { background:#eef2ff; padding:8px 12px; border-radius:999px; font-weight:600; }
    /* make dataframe container full width */
    .stDataFrame { border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------- helper functions ----------
def run_query(sql: str, params: dict = None) -> pd.DataFrame:
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


from sqlalchemy import text

def execute_statement(sql: str, params: dict = None):
    """
    Execute INSERT/UPDATE/DELETE and commit immediately.
    Raises exception on failure.
    """
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(text(sql), params or {})
            trans.commit()
        except Exception as e:
            trans.rollback()
            raise


def download_df_as_csv(df: pd.DataFrame, name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name=f"{name}.csv", mime="text/csv")

def get_table_preview(table_name: str, limit: int = 250) -> pd.DataFrame:
    return run_query(f"SELECT * FROM {table_name} ORDER BY 1 LIMIT :limit", {"limit": limit})

# --------- analytics queries mapping (predefined, safe) ----------
ANALYTICS = {
    "Count of donors per city": {
        "sql": "SELECT city, COUNT(*) AS count FROM donors GROUP BY city ORDER BY count DESC;"
    },
    "Count of collectors per city": {
        "sql": "SELECT city, COUNT(*) AS count FROM collectors GROUP BY city ORDER BY count DESC;"
    },
    "Top cities with highest ewaste donations (items count)": {
        "sql": """
            SELECT d.city, COUNT(e.waste_id) AS total_listings
            FROM ewaste_listings e
            JOIN donors d ON e.donor_id = d.donor_id
            GROUP BY d.city ORDER BY total_listings DESC;
        """
    },
    "Most common e-waste categories donated": {
        "sql": "SELECT category, COUNT(*) AS count FROM ewaste_listings GROUP BY category ORDER BY count DESC LIMIT 20;"
    },
    "Status breakdown (Available vs Picked vs Expired)": {
        "sql": "SELECT status, COUNT(*) AS count FROM ewaste_listings GROUP BY status ORDER BY count DESC;"
    },
    "Collectors with highest pickups (by number of requests accepted/completed)": {
        "sql": """
            SELECT c.name AS collector, COUNT(p.request_id) AS pickups
            FROM pickup_requests p
            JOIN collectors c ON p.collector_id = c.collector_id
            WHERE p.status IN ('Accepted','Completed')
            GROUP BY c.name ORDER BY pickups DESC LIMIT 20;
        """
    },
    "Donors with maximum contributions (listings count)": {
        "sql": """
            SELECT d.name AS donor, COUNT(e.waste_id) AS listings
            FROM ewaste_listings e
            JOIN donors d ON e.donor_id = d.donor_id
            GROUP BY d.name ORDER BY listings DESC LIMIT 20;
        """
    },
    "Pickup requests: Completed vs Pending vs Cancelled": {
        "sql": "SELECT status, COUNT(*) AS count FROM pickup_requests GROUP BY status ORDER BY count DESC;"
    },
    "Avg. quantity donated per donor": {
        "sql": "SELECT d.name, AVG(e.quantity) AS avg_qty FROM ewaste_listings e JOIN donors d ON e.donor_id=d.donor_id GROUP BY d.name ORDER BY avg_qty DESC LIMIT 20;"
    },
    "City-wise ewaste demand vs supply (supply = listings count, demand = claims count)": {
        "sql": """
            SELECT co.city,
                   COALESCE(supply,0) AS supply,
                   COALESCE(demand,0) AS demand
            FROM (
                SELECT city, COUNT(*) AS supply
                FROM donors d JOIN ewaste_listings e ON d.donor_id = e.donor_id
                GROUP BY city
            ) s
            FULL OUTER JOIN (
                SELECT d.city, COUNT(c.claim_id) AS demand
                FROM pickup_requests c
                JOIN ewaste_listings e ON c.waste_id = e.waste_id
                JOIN donors d ON e.donor_id = d.donor_id
                GROUP BY d.city
            ) q ON s.city = q.city
            -- normalize key name
            LEFT JOIN (SELECT DISTINCT city AS city FROM donors) co ON co.city = COALESCE(s.city, q.city)
            ORDER BY supply DESC NULLS LAST;
        """
    },
    "Trend of donations over time (monthly)": {
        "sql": """
            SELECT date_trunc('month', created_at)::date AS month, COUNT(*) AS listings
            FROM ewaste_listings
            GROUP BY month ORDER BY month;
        """
    },
    # add more analytics queries as desired...
}

# add also donors+collectors combined query
ANALYTICS.update({
    "Donors & Collectors counts side-by-side per city": {
        "sql": """
            SELECT COALESCE(d.city,c.city) AS city,
                   COALESCE(d.count_d,0) AS donors,
                   COALESCE(c.count_c,0) AS collectors
            FROM (SELECT city, COUNT(*) AS count_d FROM donors GROUP BY city) d
            FULL OUTER JOIN (SELECT city, COUNT(*) AS count_c FROM collectors GROUP BY city) c
            ON d.city = c.city
            ORDER BY COALESCE(d.count_d,0)+COALESCE(c.count_c,0) DESC;
        """
    }
})

# --------- UI: Tabs ---------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "View Tables", "View Analytics", "CRUD Operations", "About Me"])

# ---------- TAB 1: OVERVIEW ----------
with tab1:
    col1, col2 = st.columns([2,3])
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## ♻️ E-Waste Management System")
        st.markdown("**Real-time platform** to connect donors with collectors and track pickups, status and analytics.")
        st.markdown("<div class='muted'>Built with PostgreSQL, SQLAlchemy, Pandas, Plotly & Streamlit.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Quick actions")
        c1, c2, c3 = st.columns(3)
        if c1.button("🎯 Objective"):
            st.info("Objective: Reduce e-waste by enabling easy donations, pickups and tracking.")
        if c2.button("⚙️ How it works"):
            st.info("Donor lists item → Collector claims & schedules pickup → Admin monitors analytics.")
        if c3.button("🌱 Impact"):
            st.info("Impact: Less landfill waste, more recycling, community engagement.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        # show a few live metrics
        try:
            donors_count = run_query("SELECT COUNT(*) FROM donors;").iloc[0,0]
            collectors_count = run_query("SELECT COUNT(*) FROM collectors;").iloc[0,0]
            listings_count = run_query("SELECT COUNT(*) FROM ewaste_listings;").iloc[0,0]
            pickups_count = run_query("SELECT COUNT(*) FROM pickup_requests;").iloc[0,0]
        except Exception as e:
            donors_count = collectors_count = listings_count = pickups_count = "—"
        st.markdown("<div class='big'>Live summary</div>", unsafe_allow_html=True)
        st.write("")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Donors", donors_count)
        m2.metric("Collectors", collectors_count)
        m3.metric("Listings", listings_count)
        m4.metric("Pickup Requests", pickups_count)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- TAB 2: VIEW TABLES ----------
with tab2:
    st.header("📋 View & Download Tables")
    tables = ["donors", "collectors", "ewaste_listings", "pickup_requests"]
    left, right = st.columns([3,1])
    with right:
        table_choice = st.selectbox("Choose table", tables)
        max_rows = st.number_input("Max rows to preview", min_value=50, max_value=5000, value=500, step=50)
        search_text = st.text_input("Search (searches all text columns - case-insensitive)", "")
    with left:
        if table_choice:
            df_table = get_table_preview(table_choice, limit=int(max_rows))
            # if user search_text, apply basic filter across object columns
            if search_text.strip():
                s = search_text.strip().lower()
                # apply to string columns
                str_cols = df_table.select_dtypes(include="object").columns.tolist()
                if str_cols:
                    mask = pd.Series(False, index=df_table.index)
                    for c in str_cols:
                        mask = mask | df_table[c].astype(str).str.lower().str.contains(s)
                    df_table = df_table[mask]
            st.write(f"### Preview: `{table_choice}` — {len(df_table)} rows shown")
            st.dataframe(df_table, use_container_width=True)
            download_df_as_csv(df_table, table_choice)

            with st.expander("Show raw SQL (read-only)"):
                st.code(f"SELECT * FROM {table_choice} LIMIT {max_rows};")
        else:
            st.info("Select a table from the right panel")

# ---------- TAB 3: VIEW ANALYTICS ----------
with tab3:
    st.header("📈 Analytics — SQL powered")
    st.write("Choose an analysis from the dropdown. Results are fetched live from the database and visualized.")
    analytics_options = list(ANALYTICS.keys())
    col_a, col_b = st.columns([3,1])
    with col_a:
        selected_analytic = st.selectbox("Select analysis", analytics_options)
    with col_b:
        topn = st.number_input("Result limit (where applicable)", min_value=5, max_value=200, value=20, step=5)

    if selected_analytic:
        sql = ANALYTICS[selected_analytic]["sql"]
        # (some queries may contain LIMIT internally; we respect them. For others we add limit)
        try:
            df = run_query(sql)
            if "limit" not in sql.lower() and df.shape[0] > topn:
                df = df.head(topn)
            st.markdown("#### Query Result")
            st.dataframe(df, use_container_width=True)

            # choose type of chart heuristically
            if df.shape[1] >= 2:
                # if first column is categorical and second numeric -> bar + pie if small categories
                xcol = df.columns[0]
                ycol = df.columns[1] if pd.api.types.is_numeric_dtype(df[df.columns[1]]) else None

                if ycol is not None and pd.api.types.is_numeric_dtype(df[ycol]):
                    fig = px.bar(df, x=xcol, y=ycol, title=selected_analytic, labels={xcol: xcol, ycol: ycol})
                    st.plotly_chart(fig, use_container_width=True)
                    # pie chart if categories not too many
                    if df.shape[0] <= 20:
                        fig2 = px.pie(df, names=xcol, values=ycol, title=f"{selected_analytic} — share")
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    # fallback table / text
                    st.info("No numeric column for plotting; showing table only.")
            else:
                st.info("Result has one column only — table shown above.")
            # CSV download
            download_df_as_csv(df, f"analytics_{selected_analytic[:30].replace(' ','_')}")
        except Exception as e:
            st.error("Failed to run query: " + str(e))

# ---------- TAB 4: CRUD OPERATIONS ----------
with tab4:
    st.header("✍️ CRUD: Add / Update / Delete records")
    st.write("Use the forms to manage Donors, Collectors, E-Waste Listings and Pickup Requests.")
    crud_tabs = st.tabs(["Donors", "Collectors", "Listings", "Pickup Requests"])

    # ---------- Donors ----------
    with crud_tabs[0]:
        st.subheader("Donors")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            with st.form("add_donor"):
                st.markdown("**Add new donor**")
                dn_name = st.text_input("Name")
                dn_type = st.selectbox("Type", ["Household", "Institution", "Business"], index=0)
                dn_address = st.text_area("Address")
                dn_city = st.text_input("City")
                dn_contact = st.text_input("Contact")
                submitted = st.form_submit_button("Add Donor")
                if submitted:
                    if not dn_name.strip():
                        st.warning("Name required")
                    else:
                        sql = """
                            INSERT INTO donors (name, type, address, city, contact)
                            VALUES (:name, :type, :address, :city, :contact)
                        """
                        execute_statement(sql, {"name": dn_name.strip(), "type": dn_type, "address": dn_address.strip(),
                                                "city": dn_city.strip(), "contact": dn_contact.strip()})
                        st.success("Donor added successfully")
        with dcol2:
            st.markdown("**Modify / Delete donor**")
            donors_df = run_query("SELECT donor_id, name, city FROM donors ORDER BY name LIMIT 500;")
            if donors_df.empty:
                st.info("No donors found")
            else:
                donor_sel = st.selectbox("Select donor", donors_df.apply(lambda r: f"{r['donor_id']} - {r['name']} ({r['city']})", axis=1))
                did = int(donor_sel.split(" - ")[0])
                donor_full = run_query("SELECT * FROM donors WHERE donor_id = :id", {"id": did})
                st.write(donor_full)
                new_name = st.text_input("New name", value=donor_full.at[0,'name'])
                new_type = st.selectbox("New type", ["Household", "Institution", "Business"], index=["Household","Institution","Business"].index(donor_full.at[0,'type']) if donor_full.at[0,'type'] in ["Household","Institution","Business"] else 0)
                new_address = st.text_area("New address", value=donor_full.at[0,'address'])
                new_city = st.text_input("New city", value=donor_full.at[0,'city'])
                new_contact = st.text_input("New contact", value=str(donor_full.at[0,'contact']))
                if st.button("Update donor"):
                    sql = """
                        UPDATE donors
                        SET name = :name, type = :type, address = :address, city = :city, contact = :contact
                        WHERE donor_id = :id
                    """
                    execute_statement(sql, {"name": new_name.strip(), "type": new_type, "address": new_address.strip(),
                                            "city": new_city.strip(), "contact": new_contact.strip(), "id": did})
                    st.success("Donor updated")
                if st.button("Delete donor"):
                    # caution: deleting donor may cascade/violate fk. We will block if listings exist.
                    existing = run_query("SELECT COUNT(*) FROM ewaste_listings WHERE donor_id = :id", {"id": did}).iloc[0,0]
                    if existing > 0:
                        st.warning(f"Cannot delete donor — they have {existing} listing(s). Delete those first.")
                    else:
                        execute_statement("DELETE FROM donors WHERE donor_id = :id", {"id": did})
                        st.success("Donor deleted")

    # ---------- Collectors ----------
    with crud_tabs[1]:
        st.subheader("Collectors")
        c1, c2 = st.columns(2)
        with c1:
            with st.form("add_collector"):
                st.markdown("**Add new collector**")
                cn_name = st.text_input("Name", key="col_name")
                cn_type = st.selectbox("Type", ["NGO", "Private", "Government"], index=0, key="col_type")
                cn_address = st.text_area("Address", key="col_address")
                cn_city = st.text_input("City", key="col_city")
                cn_contact = st.text_input("Contact", key="col_contact")
                submitted_c = st.form_submit_button("Add Collector")
                if submitted_c:
                    if not cn_name.strip():
                        st.warning("Name required")
                    else:
                        execute_statement("""
                            INSERT INTO collectors (name, type, address, city, contact)
                            VALUES (:name,:type,:address,:city,:contact)
                        """, {"name": cn_name.strip(), "type": cn_type, "address": cn_address.strip(),
                              "city": cn_city.strip(), "contact": cn_contact.strip()})
                        st.success("Collector added")
        with c2:
            collectors_df = run_query("SELECT collector_id, name, city FROM collectors ORDER BY name LIMIT 500;")
            if collectors_df.empty:
                st.info("No collectors found")
            else:
                collector_sel = st.selectbox("Select collector", collectors_df.apply(lambda r: f"{r['collector_id']} - {r['name']} ({r['city']})", axis=1))
                cid = int(collector_sel.split(" - ")[0])
                cc = run_query("SELECT * FROM collectors WHERE collector_id = :id", {"id": cid})
                st.write(cc)
                new_cname = st.text_input("New name", value=cc.at[0,'name'])
                new_ctype = st.selectbox("New type", ["NGO", "Private", "Government"], index=["NGO","Private","Government"].index(cc.at[0,'type']) if cc.at[0,'type'] in ["NGO","Private","Government"] else 0)
                new_caddress = st.text_area("New address", value=cc.at[0,'address'])
                new_ccity = st.text_input("New city", value=cc.at[0,'city'])
                new_ccontact = st.text_input("New contact", value=str(cc.at[0,'contact']))
                if st.button("Update collector"):
                    execute_statement("""
                        UPDATE collectors
                        SET name=:name, type=:type, address=:address, city=:city, contact=:contact
                        WHERE collector_id=:id
                    """, {"name": new_cname.strip(), "type": new_ctype, "address": new_caddress.strip(),
                          "city": new_ccity.strip(), "contact": new_ccontact.strip(), "id": cid})
                    st.success("Collector updated")
                if st.button("Delete collector"):
                    # check pickups exist
                    existing = run_query("SELECT COUNT(*) FROM pickup_requests WHERE collector_id = :id", {"id": cid}).iloc[0,0]
                    if existing > 0:
                        st.warning(f"Cannot delete collector — they have {existing} pickup request(s).")
                    else:
                        execute_statement("DELETE FROM collectors WHERE collector_id = :id", {"id": cid})
                        st.success("Collector deleted")

    # ---------- Listings ----------
    with crud_tabs[2]:
        st.subheader("E-Waste Listings")
        L1, L2 = st.columns(2)
        with L1:
            with st.form("add_listing"):
                st.markdown("**Create new listing**")
                provs = run_query("SELECT donor_id, name FROM donors ORDER BY name LIMIT 1000;")
                if provs.empty:
                    st.warning("No donors found — add donors first")
                else:
                    prov_map = {f"{r['donor_id']} - {r['name']}": r['donor_id'] for _, r in provs.iterrows()}
                    sel_prov = st.selectbox("Select Donor", list(prov_map.keys()))
                    item_name = st.text_input("Item name")
                    category = st.text_input("Category (e.g. Mobile, Laptop, Appliance)")
                    quantity = st.number_input("Quantity", min_value=1, value=1)
                    condition = st.selectbox("Condition", ["Working", "Damaged", "Parts only", "Unknown"])
                    created_at = st.date_input("Created At", value=datetime.today())
                    expiry = st.date_input("Expiry Date (optional)", value=None)
                    status = st.selectbox("Status", ["Available", "Picked", "Expired"], index=0)
                    add_listing = st.form_submit_button("Add listing")
                    if add_listing:
                        prov_id = prov_map[sel_prov]
                        execute_statement("""
                            INSERT INTO ewaste_listings (item_name, category, quantity, condition, donor_id, created_at, expiry_date, status)
                            VALUES (:item_name, :category, :quantity, :condition, :donor_id, :created_at, :expiry_date, :status)
                        """, {
                            "item_name": item_name.strip(),
                            "category": category.strip(),
                            "quantity": int(quantity),
                            "condition": condition,
                            "donor_id": int(prov_id),
                            "created_at": datetime.combine(created_at, datetime.min.time()),
                            "expiry_date": (datetime.combine(expiry, datetime.min.time()) if expiry else None),
                            "status": status
                        })
                        st.success("Listing added")
        with L2:
            st.markdown("**Update listing status / details**")
            listings = run_query("SELECT waste_id, item_name, status FROM ewaste_listings ORDER BY created_at DESC LIMIT 1000;")
            if listings.empty:
                st.info("No listings")
            else:
                sel_listing = st.selectbox("Select listing", listings.apply(lambda r: f"{r['waste_id']} - {r['item_name']} [{r['status']}]", axis=1))
                wid = int(sel_listing.split(" - ")[0])
                lrow = run_query("SELECT * FROM ewaste_listings WHERE waste_id=:id", {"id": wid})
                st.write(lrow)
                new_item = st.text_input("Item name", value=lrow.at[0,'item_name'])
                new_cat = st.text_input("Category", value=lrow.at[0,'category'])
                new_qty = st.number_input("Quantity", min_value=0, value=int(lrow.at[0,'quantity']))
                new_cond = st.selectbox("Condition", ["Working","Damaged","Parts only","Unknown"], index=["Working","Damaged","Parts only","Unknown"].index(lrow.at[0,'condition']) if lrow.at[0,'condition'] in ["Working","Damaged","Parts only","Unknown"] else 3)
                new_status = st.selectbox("Status", ["Available","Picked","Expired"], index=["Available","Picked","Expired"].index(lrow.at[0,'status']) if lrow.at[0,'status'] in ["Available","Picked","Expired"] else 0)
                if st.button("Update listing"):
                    execute_statement("""
                        UPDATE ewaste_listings
                        SET item_name=:item_name, category=:category, quantity=:quantity, condition=:condition, status=:status
                        WHERE waste_id=:id
                    """, {"item_name": new_item.strip(), "category": new_cat.strip(), "quantity": int(new_qty), "condition": new_cond, "status": new_status, "id": wid})
                    st.success("Listing updated")
                if st.button("Delete listing"):
                    # also delete pickup requests referencing it
                    execute_statement("DELETE FROM pickup_requests WHERE waste_id = :id", {"id": wid})
                    execute_statement("DELETE FROM ewaste_listings WHERE waste_id = :id", {"id": wid})
                    st.success("Listing and related pickup requests deleted")

    # ---------- Pickup Requests ----------
    with crud_tabs[3]:
        st.subheader("Pickup Requests")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            with st.form("create_pickup"):
                st.markdown("**Create / Request pickup** (Collector selects an available listing)")
                collectors = run_query("SELECT collector_id, name FROM collectors ORDER BY name LIMIT 1000;")
                listings_avail = run_query("SELECT waste_id, item_name FROM ewaste_listings WHERE status='Available' ORDER BY created_at LIMIT 1000;")
                if collectors.empty or listings_avail.empty:
                    st.warning("Need at least one collector AND one available listing to create a pickup.")
                else:
                    coll_map = {f"{r['collector_id']} - {r['name']}": r['collector_id'] for _, r in collectors.iterrows()}
                    list_map = {f"{r['waste_id']} - {r['item_name']}": r['waste_id'] for _, r in listings_avail.iterrows()}
                    sel_col = st.selectbox("Collector", list(coll_map.keys()))
                    sel_list = st.selectbox("Listing", list(list_map.keys()))
                    pick_status = st.selectbox("Initial status", ["Requested","Accepted","Completed","Cancelled"])
                    create_pick = st.form_submit_button("Create pickup request")
                    if create_pick:
                        execute_statement("""
                            INSERT INTO pickup_requests (waste_id, collector_id, status, timestamp)
                            VALUES (:waste_id, :collector_id, :status, :ts)
                        """, {"waste_id": list_map[sel_list], "collector_id": coll_map[sel_col], "status": pick_status, "ts": datetime.now()})
                        # optionally update listing status when accepted/completed
                        if pick_status in ("Accepted","Completed"):
                            execute_statement("UPDATE ewaste_listings SET status='Picked' WHERE waste_id=:id", {"id": list_map[sel_list]})
                        st.success("Pickup request created")
        with pcol2:
            st.markdown("**Manage existing pickup requests**")
            pr = run_query("SELECT request_id, status, timestamp FROM pickup_requests ORDER BY timestamp DESC LIMIT 200;")
            if pr.empty:
                st.info("No pickup requests")
            else:
                sel_pr = st.selectbox("Select pickup", pr.apply(lambda r: f"{r['request_id']} - {r['status']} @ {r['timestamp']}", axis=1))
                pid = int(sel_pr.split(" - ")[0])
                prow = run_query("SELECT * FROM pickup_requests WHERE request_id=:id", {"id": pid})
                st.write(prow)
                new_status = st.selectbox("Update status", ["Requested","Accepted","Completed","Cancelled"])
                if st.button("Update pickup status"):
                    execute_statement("UPDATE pickup_requests SET status=:status WHERE request_id=:id", {"status": new_status, "id": pid})
                    st.success("Pickup request status updated")
                    # if completed -> update listing status
                    if new_status == "Completed":
                        wid = int(prow.at[0,'waste_id'])
                        execute_statement("UPDATE ewaste_listings SET status='Picked' WHERE waste_id=:id", {"id": wid})

    

# ---------- TAB 5: ABOUT ME ----------
with tab5:
    st.header("About the project & Author")
    st.markdown("""
    **Project:** E-Waste Management System — a small real-time app to connect donors with collectors and reduce electronic waste.
    
    **Features implemented here:**
    - Live table viewer + CSV download
    - 15+ Analytics queries with bar/pie charts
    - CRUD for donors, collectors, listings, pickup requests
    - Bulk maintenance actions (mark expired, cleanup)
    
    **Tech:** Python, Streamlit, PostgreSQL, SQLAlchemy, Pandas, Plotly.
    """)
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown("### About Me")
        st.markdown("""
        - MCA Student | Data Science Enthusiast  
        - Passionate about building socially-impactful data apps  
        - GitHub: https://github.com/your-username  (update your link)
        - LinkedIn: https://linkedin.com/in/your-profile  (update your link)
        """)
    with col2:
        st.image("https://images.unsplash.com/photo-1526378723343-0f44bb6a8a79?auto=format&fit=crop&w=600&q=60", caption="Recycle & Reuse", use_column_width=True)

    st.markdown("---")
    st.markdown("**License:** MIT. Feel free to adapt & reuse with attribution.")

# ---------- Footer ----------
st.markdown("<div style='padding:16px 0; text-align:center; color:#6b7280;'>Built with ❤️ — Streamlit + PostgreSQL</div>", unsafe_allow_html=True)

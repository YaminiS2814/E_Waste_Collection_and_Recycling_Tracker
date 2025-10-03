# E_Waste_Collection_and_Recycling_Tracker
A real-time, web application for managing e-waste donations and collections using Python, Streamlit, and PostgreSQL.

## 🚀 Project Overview
This project aims to reduce electronic waste by connecting e-waste Donors (households, businesses, institutions) with Collectors (recycling centers, NGOs, agencies). It uses data-driven insights to track donations, optimize pickups, and promote sustainability.

## 🌱 Key Goals
- Minimize e-waste accumulation through digital donation tracking.
- Support responsible recycling efforts by streamlining pickup requests.
- Enable transparency using interactive dashboards & SQL analysis.

## 📊 Features
- 🔍 **Live Table Viewer:** View and filter records from Donors, Collectors, Listings, and Pickups.  
- 📈 **Data Insights:** Answer 15+ SQL-powered real-world questions with visualizations (bar, pie, donut, etc.).  
- 🧠 **EDA & Analytics:** Understand trends in e-waste type, pickup status, and donor contributions.  
- ➕ **CRUD Operations:** Add, update, or delete records via user-friendly forms.  
- 📍 **Location Filtering:** Filter donations based on city, donor type, and e-waste category.  
- 📞 **Direct Contact:** View contact info for all registered donors and collectors.  
- 🖼 **Beautiful UI:** Styled with custom CSS and animated cards for a polished user experience.

## 📂 Technologies Used

| Tech       | Purpose                                   |
|------------|-------------------------------------------|
| Python     | Backend scripting and logic               |
| Streamlit  | Frontend app and interactivity            |
| PostgreSQL | Database for storing all e-waste records |
| Pandas     | Data manipulation and cleaning            |
| Plotly     | Visualization & analytics                 |
| dotenv     | Secure handling of DB credentials         |

## 🗂 Database Schema
- **Donors:** Donor_ID, Name, Type, Address, City, Contact  
- **Collectors:** Collector_ID, Name, Type, City, Contact  
- **Listings:** Listing_ID, E_Waste_Type, Quantity, Expiry_Date, Donor_ID, etc.  
- **Pickups:** Pickup_ID, Listing_ID, Collector_ID, Status, Timestamp  

## ❓ SQL-Based Analytical Questions Answered
- Count of donors & collectors per city  
- Top cities with highest e-waste donations  
- Most common e-waste categories donated  
- Status breakdown (Available vs Picked vs Expired)  
- Collectors with highest pickups  
- Donors with maximum contributions  
- Pickup requests: Completed vs Pending vs Cancelled  
- Avg. quantity donated per donor  
- City-wise e-waste demand vs supply  
- Trend of donations over time  
- Top donors by contribution  
- Categories with highest pickup counts  
- Collector efficiency metrics  
- Total quantity donated per donor  
- Donation trends per city  


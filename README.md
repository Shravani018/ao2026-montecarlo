# 🏆 AO2026-MonteCarlo  
**🎾 Monte Carlo Prediction of the 2026 Australian Open Champion**

---

## 📌 Overview
This project estimates each player’s probability of winning the **2026 Australian Open** using **Monte Carlo simulation** based on historical **ATP match-level data**.

The current stage focuses on building a **clean, hard-court–specific dataset** that accurately reflects Australian Open conditions and is suitable for probabilistic modeling.

---

## 📊 Data Source
- 🎾 ATP match-level data (2017–2024)
- 🌐 Public dataset maintained by Jeff Sackmann
- 📁 Yearly match files: [https://github.com/JeffSackmann/tennis_atp]

---

## ✅ Work Completed So Far

### 📥 Data Collection
- Combined ATP match data from **2017–2024** into a unified dataset.
- Standardized columns and formats across seasons.

### 🏟️ Surface Filtering
- Filtered matches to **hard courts only**, aligning with Australian Open conditions.

## 🔍 Exploratory Data Analysis (EDA)

### 🧹 Data Cleaning
- Consolidated multi-season ATP match data
- Parsed tournament dates and standardized formats
- Normalized seed and entry fields
- Forward-filled missing player rankings

### 🧠 Feature Engineering
- Constructed continuous ranking histories for winners and losers
- Encoded unseeded players explicitly

### 📊 Visualization
- Generated multiple exploratory plots to assess ranking distributions and data coverage.
---

## 🚧 Current Project Status
- ✅ Historical ATP data collated  
- ✅ Hard-court–specific dataset prepared  
- ✅ Cleaned and structured data ready for modeling  
- ⏳ Win probability modeling  
- ⏳ Monte Carlo tournament simulation  

---

## 🔜 Next Steps
- 📈 Develop match-level win probability models (ranking-based / Elo-style).
- 🔁 Simulate full Australian Open tournament draws using Monte Carlo methods.
- 🧮 Aggregate simulation outputs into player win probabilities.
- 📊 Visualize and interpret outcome distributions.

---

## ⚠️ Disclaimer
All results will be **probabilistic simulations**, not deterministic predictions.

---

## 👤 Author
Built as a personal data science project exploring **sports analytics and simulation modeling**.



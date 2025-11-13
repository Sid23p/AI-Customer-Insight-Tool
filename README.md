# AI-Powered Customer Segmentation Dashboard

**Phase 3: Interactive Dashboard with Comparative Clustering Analysis**

A production-ready customer segmentation solution using RFM analysis with K-Means, DBSCAN, and K-Medoids clustering algorithms. Features an interactive Streamlit dashboard that runs 24/7 as a system service.

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Install dependencies
./setup.sh

# 2. Install as system service (runs 24/7 automatically)
./install_service.sh

# 3. Access dashboard
open http://localhost:8501
```

**That's it!** The dashboard will now run continuously, automatically restarting on crashes and surviving system reboots.

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

---

## ✨ Features

### Clustering Algorithms
- **K-Means**: Fast, efficient clustering for well-separated groups
- **DBSCAN**: Density-based clustering that handles outliers and arbitrary shapes
- **K-Medoids**: Robust clustering using actual data points as centers

### Dashboard Features
- Interactive 3D and 2D scatter plot visualizations
- Real-time algorithm switching and parameter adjustment
- Cluster summary tables with RFM metrics
- Silhouette score evaluation
- Export capabilities for segmented customer data

### System Features
- **24/7 Operation**: Runs continuously as macOS LaunchAgent
- **Auto-Recovery**: Automatically restarts on crashes
- **Boot Persistence**: Starts automatically on system boot/login
- **Zero Maintenance**: Requires no manual intervention

---

## 📦 Installation

### Prerequisites
- macOS (tested on macOS 13+)
- Python 3.9+ (automatically installed via setup script)
- Terminal access

### Step-by-Step Installation

#### Option 1: Automated One-Command Setup
```bash
./RUN_PROJECT.sh
```

#### Option 2: Manual Setup

**Step 1: Install Dependencies**
```bash
./setup.sh
```
This creates a virtual environment and installs:
- streamlit, plotly, numpy, pandas
- scikit-learn, seaborn, matplotlib
- scikit-learn-extra (for K-Medoids)

**Step 2: Install System Service**
```bash
./install_service.sh
```
This installs the dashboard as a macOS LaunchAgent that:
- Starts automatically on boot
- Restarts automatically on crashes
- Runs in the background

**Step 3: Verify Installation**
```bash
./status.sh
```

---

## 💻 Usage

### Accessing the Dashboard

Once installed, the dashboard is always available at:
```
http://localhost:8501
```

### Using the Dashboard

1. **Select Algorithm**: Choose K-Means, DBSCAN, or K-Medoids from the sidebar
2. **Adjust Parameters**:
   - K-Means/K-Medoids: Set number of clusters (k) with slider
   - DBSCAN: Adjust `eps` and `min_samples` parameters
3. **Explore Results**: 
   - View 3D and 2D cluster visualizations
   - Review cluster summaries and metrics
   - Export segmented data

### Management Commands

```bash
# Check dashboard status
./status.sh

# View live logs
tail -f dashboard_service.log

# View error logs
tail -f dashboard_service_error.log

# Uninstall service (if needed)
./uninstall_service.sh
```

### Manual Mode (Alternative)

If you prefer to run manually instead of as a service:

```bash
# Start dashboard manually
./start_dashboard.sh

# Check status
./status.sh

# Stop dashboard
./stop_dashboard.sh
```

---

## 📁 Project Structure

```
.
├── app_dashboard.py              # Main Streamlit dashboard application
├── segmentation_phase3.py         # Clustering algorithms and utilities
├── customer_segmentation_v2.py   # Original Phase 1 & 2 implementation
│
├── setup.sh                      # Dependency installation script
├── install_service.sh            # System service installation
├── uninstall_service.sh          # Service removal
├── fix_service.sh                # Service repair utility
├── status.sh                     # Unified status checker
│
├── start_dashboard.sh            # Manual start script
├── stop_dashboard.sh             # Manual stop script
├── start_dashboard_service.sh    # Service wrapper (used by launchd)
│
├── RUN_PROJECT.sh                # One-command setup script
│
├── online_retail.csv             # Input data file
├── segmented_customers_*.csv     # Generated segmentation results
│
├── dashboard_service.log         # Service runtime logs
├── dashboard_service_error.log   # Service error logs
│
└── README.md                     # This file
```

### Key Files

- **`app_dashboard.py`**: Interactive Streamlit dashboard
- **`segmentation_phase3.py`**: Clustering algorithms (K-Means, DBSCAN, K-Medoids)
- **`setup.sh`**: Automated dependency installation
- **`install_service.sh`**: System service installation
- **`status.sh`**: Unified status checker

---

## 🔧 Troubleshooting

### Dashboard Not Accessible

**Check status:**
```bash
./status.sh
```

**Check if port is listening:**
```bash
lsof -i :8501
```

**View error logs:**
```bash
tail -20 dashboard_service_error.log
```

**Fix service:**
```bash
./fix_service.sh
```

### Service Not Starting

1. **Uninstall and reinstall:**
   ```bash
   ./uninstall_service.sh
   ./install_service.sh
   ```

2. **Check virtual environment:**
   ```bash
   ls -la .venv/bin/python*
   ```

3. **Test manually:**
   ```bash
   source .venv/bin/activate
   streamlit run app_dashboard.py
   ```

### Common Issues

**"Operation not permitted" errors:**
- Run `./fix_service.sh` to apply the latest fixes
- Ensure Terminal has necessary permissions in System Settings

**Port 8501 already in use:**
```bash
lsof -ti :8501 | xargs kill -9
./install_service.sh
```

**Dependencies not found:**
```bash
./setup.sh
```

---

## 📚 Documentation

- **This README**: Main project documentation
- **Quick Start Guide**: See `QUICK_START_GUIDE.md` for detailed step-by-step instructions
- **Fix Guide**: See `FIX_CONNECTION_ERROR.md` for troubleshooting connection issues

---

## 🎯 System Service Details

### Service Configuration

- **Service Name**: `com.customer-segmentation.dashboard`
- **Plist Location**: `~/Library/LaunchAgents/com.customer-segmentation.dashboard.plist`
- **Auto-Start**: Enabled (starts on boot/login)
- **Auto-Restart**: Enabled (restarts on crashes)
- **Port**: 8501
- **Logs**: `dashboard_service.log` and `dashboard_service_error.log`

### Service Management

```bash
# Check if service is loaded
launchctl list | grep customer-segmentation

# Manually start service
launchctl load ~/Library/LaunchAgents/com.customer-segmentation.dashboard.plist

# Manually stop service
launchctl unload ~/Library/LaunchAgents/com.customer-segmentation.dashboard.plist
```

---

## 🔬 Technical Details

### Clustering Algorithms

- **K-Means**: Uses scikit-learn's KMeans with optimal k=3 (determined via elbow method and silhouette analysis)
- **DBSCAN**: Density-based clustering with configurable `eps` and `min_samples`
- **K-Medoids**: Uses scikit-learn-extra's KMedoids for robust clustering

### Data Processing

- **RFM Metrics**: Recency, Frequency, Monetary value calculation
- **Preprocessing**: Log transformation for Monetary, StandardScaler normalization
- **Output**: CSV files with cluster assignments for each customer

### Performance

- **Startup Time**: ~5-10 seconds
- **Memory Usage**: ~200-500 MB (depending on data size)
- **Response Time**: <1 second for algorithm switching

---

## 📊 Output Files

- `segmented_customers_kmeans.csv`: K-Means clustering results
- `segmented_customers_dbscan.csv`: DBSCAN clustering results  
- `segmented_customers_kmedoids.csv`: K-Medoids clustering results

Each CSV contains:
- CustomerID
- Recency, Frequency, Monetary values
- Cluster assignment

---

## 🛠️ Development

### Running in Development Mode

```bash
source .venv/bin/activate
streamlit run app_dashboard.py
```

### Adding New Algorithms

1. Add clustering function to `segmentation_phase3.py`
2. Add option to dashboard sidebar in `app_dashboard.py`
3. Update service scripts if needed

---

## 📝 License

This project is part of a Master's degree program in Computer Science.

---

## 👤 Author

Siddhant Patil - 2025

---

## 🙏 Acknowledgments

- Streamlit for the dashboard framework
- scikit-learn for clustering algorithms
- Plotly for interactive visualizations

---

**Last Updated**: 2025  
**Version**: Phase 3 - Production Release

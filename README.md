# AI-Driven PPA Regression Infrastructure for IP Evaluation

An end-to-end, production-grade flow to automate Power, Performance, and Area (PPA) regression for various RTL IPs using Synopsys tools and machine learning. The infrastructure integrates Design Compiler, IC Compiler II, and PrimeTime with Python-based AI models for predictive analytics and outlier detection.

## Table of Contents
- Overview
- Tools and Technologies
- Infrastructure Architecture
- Project Structure
- Detailed Flow Execution
- AI PPA Prediction Model
- Key TCL Scripts
- Python Infrastructure Scripts
- Folder Tree
- Author Info

---

## Overview
This infrastructure is designed to:
- Perform RTL-to-GDSII flow on multiple IPs across corners/configs
- Extract and log detailed PPA metrics from Synopsys reports
- Train machine learning models on PPA metrics
- Predict PPA of unseen RTLs and detect outliers automatically

---

## Tools and Technologies

| Task              | Tool/Technology           |
|------------------|---------------------------|
| RTL Synthesis     | Synopsys Design Compiler  |
| Place & Route     | Synopsys IC Compiler II   |
| STA & Power       | Synopsys PrimeTime        |
| Automation & AI   | Python, scikit-learn, pandas, numpy |
| Data Storage      | CSV, Pickle               |

---

## Infrastructure Architecture

```
ppa_ai_regression/
├── infra/
│   ├── runner.py         # Manages end-to-end RTL to GDSII regression
│   ├── extractor.py      # Parses area, timing, and power reports
│   └── predictor.py      # Loads ML model and performs PPA prediction
├── ml/
│   ├── train_model.py    # Trains model from collected metrics
│   └── models.pkl        # Serialized ML models
├── datasets/
│   └── ppa_metrics.csv   # Aggregated metrics from runs
├── scripts/
│   ├── dc.tcl            # Design Compiler synthesis
│   ├── icc2.tcl          # ICC2 Place & Route
│   └── pt.tcl            # PrimeTime analysis
├── rtl/
│   └── multiple_ip_*.v   # IP RTL files
├── constraints/
│   └── *.sdc             # SDC constraints for each IP
├── reports/
│   ├── synthesis/
│   ├── layout/
│   └── timing/
├── doc/
│   └── architecture.png, loss_curve.png
└── run_config.json       # List of IPs and configurations
```

---

## Detailed Flow Execution

1. **Prepare Configuration:**
   - `run_config.json` contains IP names and paths to RTL + SDC

2. **Automated Flow Execution:**
   - Run the end-to-end flow:
     ```bash
     python3 infra/runner.py --config run_config.json
     ```
   - Executes DC → ICC2 → PT for each IP
   - Generates area, power, and timing reports per IP

3. **Report Parsing:**
   - Extract metrics:
     ```bash
     python3 infra/extractor.py --report_dir reports/ --out datasets/ppa_metrics.csv
     ```

4. **Model Training:**
   - Train the prediction engine:
     ```bash
     python3 ml/train_model.py datasets/ppa_metrics.csv
     ```

5. **Prediction on New IPs:**
   - Use trained model to estimate PPA:
     ```bash
     python3 infra/predictor.py --rtl rtl/new_ip.v --sdc constraints/new_ip.sdc
     ```

---

## AI PPA Prediction Model

- **Inputs (Features):**
  - Gate Count, Flip-Flop Count, Mux Ratio, Tech Node, Clock Frequency

- **Outputs (Labels):**
  - Area (mm²), Total Power (mW), Timing Slack (ns)

- **ML Models Used:**
  - Linear Regression, Random Forest, Gradient Boosting

- **Evaluation Metrics:**
  - MAE (Mean Absolute Error), R² Score

---

## Key TCL Scripts

**scripts/dc.tcl**
```tcl
set IP [getenv "IP"]
read_verilog ../rtl/${IP}.v
read_sdc ../constraints/${IP}.sdc
set_top ${IP}
link
compile_ultra
report_area > ../reports/synthesis/${IP}_area.rpt
report_power > ../reports/synthesis/${IP}_power.rpt
report_timing > ../reports/synthesis/${IP}_timing.rpt
write -format ddc -output ../synth/${IP}.ddc
```

**scripts/icc2.tcl**
```tcl
set IP [getenv "IP"]
read_ddc ../synth/${IP}.ddc
create_floorplan -core_utilization 0.6 -aspect_ratio 1 -row_height 2 -site core
place_opt
clock_opt
route_opt
write_parasitics -spef ../reports/layout/${IP}.spef
report_timing > ../reports/layout/${IP}_timing.rpt
```

**scripts/pt.tcl**
```tcl
set IP [getenv "IP"]
read_verilog ../rtl/${IP}.v
read_sdc ../constraints/${IP}.sdc
read_parasitics ../reports/layout/${IP}.spef
report_power > ../reports/timing/${IP}_pt_power.rpt
report_timing > ../reports/timing/${IP}_pt_timing.rpt
```

---

## Python Infrastructure Scripts

**infra/runner.py**
```python
import json, os, subprocess
with open('run_config.json') as f:
    configs = json.load(f)
for ip in configs['ips']:
    os.environ['IP'] = ip['name']
    subprocess.run(['dc_shell', '-f', 'scripts/dc.tcl'])
    subprocess.run(['icc2_shell', '-f', 'scripts/icc2.tcl'])
    subprocess.run(['pt_shell', '-f', 'scripts/pt.tcl'])
```

**infra/extractor.py**
```python
import csv, re, os
report_dir = 'reports/'
out_file = 'datasets/ppa_metrics.csv'
with open(out_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['IP', 'Area', 'Power', 'Slack'])
    for ip in os.listdir(report_dir + 'synthesis/'):
        name = ip.split('_')[0]
        area = float(re.search(r'Total cell area: (\d+\.\d+)', open(report_dir + f'synthesis/{name}_area.rpt').read()).group(1))
        power = float(re.search(r'Total Power = (\d+\.\d+)', open(report_dir + f'synthesis/{name}_power.rpt').read()).group(1))
        slack = float(re.search(r'slack \(VIOLATED\):\s*(-?\d+\.\d+)', open(report_dir + f'synthesis/{name}_timing.rpt').read()).group(1))
        writer.writerow([name, area, power, slack])
```

**infra/predictor.py**
```python
import pickle
import numpy as np
model = pickle.load(open('ml/models.pkl', 'rb'))
features = np.array([[25000, 12000, 0.4, 7, 1.0]])  # example
prediction = model.predict(features)
print('Predicted Area:', prediction[0][0])
print('Predicted Power:', prediction[0][1])
print('Predicted Slack:', prediction[0][2])
```

**ml/train_model.py**
```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle
csv_path = 'datasets/ppa_metrics.csv'
data = pd.read_csv(csv_path)
X = data[['Gate_Count', 'FF_Count', 'Mux_Ratio', 'Tech_Node', 'Clock']]
y = data[['Area', 'Power', 'Slack']]
model = MultiOutputRegressor(RandomForestRegressor()).fit(X, y)
pickle.dump(model, open('ml/models.pkl', 'wb'))
```

---

## Folder Tree
```
ppa_ai_regression/
├── constraints/        # SDC per IP
├── datasets/           # CSV logs for training
├── doc/                # Flow diagrams and plots
├── infra/              # Python automation framework
├── ml/                 # ML model training and inference
├── reports/            # All tool outputs
├── rtl/                # IP RTLs
├── scripts/            # TCL automation
└── README.md
```

---

## Author
Adarsh Prakash  
Email: kumaradarsh663@gmail.com  
LinkedIn: https://linkedin.com/in/adarshprakash


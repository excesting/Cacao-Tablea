import os
import cv2
import base64
import numpy as np
import pandas as pd
from datetime import datetime, date
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

# --- IMPORT YOLO ---
try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️ Ultralytics not found. YOLO will be disabled.")
    YOLO = None

# --- IMPORT PYTORCH FORECASTING ---
try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch Forecasting not found. TFT will be disabled.")
    PYTORCH_AVAILABLE = False

# ============================================================
# 1. INITIALIZE FLASK APP & DATABASE
# ============================================================
app = Flask(__name__, static_folder='static', template_folder='templates')

# Fetch the database URL from Railway's environment variables. 
# If it doesn't exist (e.g., running locally), fallback to SQLite.
db_url = os.environ.get("DATABASE_URL", "sqlite:///tableascan.db")

# SQLAlchemy 1.4+ requires 'postgresql://' instead of 'postgres://'
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- DATABASE MODELS ---
class ScanLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    total_scanned = db.Column(db.Integer, nullable=False)
    good_pcs = db.Column(db.Integer, nullable=False)
    crack_pcs = db.Column(db.Integer, default=0)
    bloom_pcs = db.Column(db.Integer, default=0)
    defect_pcs = db.Column(db.Integer, default=0)
    yield_percentage = db.Column(db.Float, nullable=False)
    cacao_used_kg = db.Column(db.Float, nullable=False)

    def __init__(self, total_scanned, good_pcs, crack_pcs, bloom_pcs, defect_pcs):
        self.total_scanned = total_scanned
        self.good_pcs = good_pcs
        self.crack_pcs = crack_pcs
        self.bloom_pcs = bloom_pcs
        self.defect_pcs = defect_pcs
        self.yield_percentage = (good_pcs / total_scanned * 100) if total_scanned > 0 else 0.0
        # UPDATED: Changed from 215.0 to 117.0 based on your new factory requirements
        self.cacao_used_kg = (total_scanned / 117.0) * 1.5

class DailyProduction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, unique=True)
    week_id = db.Column(db.String(20), nullable=False)
    month_name = db.Column(db.String(20), nullable=False)
    
    sales_pcs = db.Column(db.Integer, default=0) 
    total_produced = db.Column(db.Integer, default=0)
    total_defects = db.Column(db.Integer, default=0)
    cracked_count = db.Column(db.Integer, default=0)
    bloom_count = db.Column(db.Integer, default=0)
    good_count = db.Column(db.Integer, default=0)
    net_usable_output = db.Column(db.Integer, default=0)

class ProductionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time_idx = db.Column(db.Integer, nullable=False, unique=True)
    week_id = db.Column(db.String(20), nullable=False, default="Unknown") # For matching logic
    branch = db.Column(db.String(50), nullable=False)
    month = db.Column(db.String(20), nullable=False)
    event_name = db.Column(db.String(100), default="None") # For special events/promos
    sales_pcs = db.Column(db.Integer, nullable=False, default=0)
    total_produced = db.Column(db.Integer, nullable=False, default=0)
    net_usable_output = db.Column(db.Integer, nullable=False, default=0)
    defect_rate = db.Column(db.Float, nullable=False, default=0.0)

with app.app_context():
    db.create_all()

# ============================================================
# 2. INITIALIZE AI MODELS
# ============================================================
yolo_model = None
tft_model = None
tft_dataset = None

if YOLO is not None:
    try:
        yolo_model = YOLO("best.pt")
        print("✅ YOLO Model Loaded")
    except Exception as e:
        print(f"⚠️ YOLO Model Error: {e}")

if PYTORCH_AVAILABLE:
    original_load = torch.load
    try:
        def global_safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            kwargs['map_location'] = torch.device('cpu')
            return original_load(*args, **kwargs)

        torch.load = global_safe_load
        tft_dataset = TimeSeriesDataSet.load("tft_dataset_v4.pkl")

        def surgical_load(*args, **kwargs):
            kwargs['weights_only'] = False
            kwargs['map_location'] = torch.device('cpu')
            ckpt = original_load(*args, **kwargs)
            if isinstance(ckpt, dict) and 'hyper_parameters' in ckpt:
                if 'loss' in ckpt['hyper_parameters']:
                    ckpt['hyper_parameters']['loss'] = QuantileLoss()
                if 'logging_metrics' in ckpt['hyper_parameters']:
                    ckpt['hyper_parameters']['logging_metrics'] = nn.ModuleList([])
            return ckpt

        torch.load = surgical_load
        tft_model = TemporalFusionTransformer.load_from_checkpoint("tft_model_v4.ckpt")
        tft_model.eval()
        torch.load = original_load
        print("✅ TFT Forecast Model Loaded Successfully!")
    except Exception as e:
        print(f"⚠️ TFT Model Error: {e}")
        torch.load = original_load

# ============================================================
# 3. HELPER: DUMMY FORECAST FALLBACK
# ============================================================
def _dummy_forecast(reason):
    return jsonify({
        "labels": ["Next Wk 1", "Next Wk 2", "Next Wk 3", "Next Wk 4"],
        "expected_demand": [45000, 52000, 60000, 48000],
        "projected_supply": [46000, 48000, 50000, 50000],
        "historical_last_4_weeks": [44000, 43000, 45000, 46000],
        "historical_defects": [0.04, 0.05, 0.03, 0.06, 0.05, 0.04, 0.02, 0.03],
        "historical_time": ["Wk 1", "Wk 2", "Wk 3", "Wk 4", "Wk 5", "Wk 6", "Wk 7", "Wk 8"],
        "status": f"Fallback: {reason}"
    })

# ============================================================
# 4. PAGE ROUTES
# ============================================================
@app.route('/')
def home():
    return render_template('home.html', current_page='home')

@app.route('/detect')
def detect():
    return render_template('detect.html', current_page='detect')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', current_page='analytics')

@app.route('/logs')
def logs():
    recent_logs = DailyProduction.query.order_by(DailyProduction.date.desc()).limit(30).all()
    weekly_logs = ProductionHistory.query.order_by(ProductionHistory.time_idx.desc()).limit(8).all()

    today = datetime.utcnow().date()
    start_of_day = datetime.combine(today, datetime.min.time())
    end_of_day = datetime.combine(today, datetime.max.time())

    today_scans = ScanLog.query.filter(
        ScanLog.timestamp >= start_of_day,
        ScanLog.timestamp <= end_of_day
    ).all()

    today_produced = sum(s.total_scanned for s in today_scans)
    today_good = sum(s.good_pcs for s in today_scans)
    today_crack = sum(s.crack_pcs for s in today_scans)
    today_bloom = sum(s.bloom_pcs for s in today_scans)
    today_defects = sum(s.crack_pcs + s.bloom_pcs + s.defect_pcs for s in today_scans)

    return render_template(
        'logs.html',
        current_page='logs',
        logs=recent_logs,
        weekly_logs=weekly_logs,
        today_produced=today_produced,
        today_good=today_good,
        today_crack=today_crack,
        today_bloom=today_bloom,
        today_defects=today_defects
    )

# ============================================================
# 5. DATA MANAGEMENT ROUTES
# ============================================================
@app.route('/api/add_log', methods=['POST'])
def add_log():
    try:
        date_str = request.form.get('date')
        produced = int(request.form.get('total_produced', 0))
        defects = int(request.form.get('total_defects', 0))
        cracked = int(request.form.get('cracked_count', 0))
        bloom = int(request.form.get('bloom_count', 0))
        good = int(request.form.get('good_count', 0))
        net_usable = good

        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        year, week_num, _ = date_obj.isocalendar()
        week_id = f"{year}-W{week_num}"
        month_name = date_obj.strftime('%B')

        daily_log = DailyProduction.query.filter_by(date=date_obj).first()
        if not daily_log:
            daily_log = DailyProduction(date=date_obj, week_id=week_id, month_name=month_name)
            db.session.add(daily_log)

        daily_log.total_produced = produced
        daily_log.total_defects = defects
        daily_log.cracked_count = cracked
        daily_log.bloom_count = bloom
        daily_log.good_count = good
        daily_log.net_usable_output = net_usable
        db.session.commit()

        # Weekly Roll-Up
        week_logs = DailyProduction.query.filter_by(week_id=week_id).all()
        w_produced = sum(l.total_produced for l in week_logs)
        w_defects = sum(l.total_defects for l in week_logs)
        w_usable = sum(l.net_usable_output for l in week_logs)
        w_defect_rate = (w_defects / w_produced) if w_produced > 0 else 0.0

        history = ProductionHistory.query.filter_by(week_id=week_id).first()
        if not history:
            max_idx = db.session.query(db.func.max(ProductionHistory.time_idx)).scalar() or 0
            history = ProductionHistory(
                time_idx=max_idx + 1, branch="Lipa", month=month_name, 
                week_id=week_id, event_name="None", sales_pcs=0
            )
            db.session.add(history)

        history.total_produced = w_produced
        history.net_usable_output = w_usable
        history.defect_rate = w_defect_rate
        db.session.commit()

        return redirect(url_for('logs'))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/add_sales', methods=['POST'])
def add_sales():
    try:
        date_str = request.form.get('sales_date')
        sales = int(request.form.get('weekly_sales', 0))

        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        year, week_num, _ = date_obj.isocalendar()
        week_id = f"{year}-W{week_num}"
        month_name = date_obj.strftime('%B')

        history = ProductionHistory.query.filter_by(week_id=week_id).first()
        if not history:
            max_idx = db.session.query(db.func.max(ProductionHistory.time_idx)).scalar() or 0
            history = ProductionHistory(
                time_idx=max_idx + 1, branch="Lipa", month=month_name, 
                week_id=week_id, event_name="None", total_produced=0, net_usable_output=0, defect_rate=0.0
            )
            db.session.add(history)

        history.sales_pcs = sales
        db.session.commit()

        return redirect(url_for('logs'))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ============================================================
# 6. API ROUTES
# ============================================================
@app.route('/api/stats')
def get_stats():
    try:
        # 1. Fetch the last 4 weeks of Weekly AI Records for the Yield calculation
        # Using net_usable_output / total_produced from history provides the weekly yield trend.
        recent_history = ProductionHistory.query.order_by(ProductionHistory.time_idx.desc()).limit(4).all()
        
        if recent_history:
            total_produced = sum(w.total_produced for w in recent_history)
            total_usable = sum(w.net_usable_output for w in recent_history)
            
            # Use 4-week rolling average
            current_yield = (total_usable / total_produced * 100) if total_produced > 0 else 0.0
            total_scanned = total_produced # Displaying total of last 4 weeks
            good_pcs = total_usable
        else:
            # Fallback if history is empty
            current_yield = 0.0
            total_scanned = 0
            good_pcs = 0

        # We still pull raw defect counts from ScanLog for the daily view (Detect page)
        cracks = db.session.query(db.func.sum(ScanLog.crack_pcs)).scalar() or 0
        bloom = db.session.query(db.func.sum(ScanLog.bloom_pcs)).scalar() or 0
        cacao_used = db.session.query(db.func.sum(ScanLog.cacao_used_kg)).scalar() or 0.0

        return jsonify({
            "total_scanned": int(total_scanned),
            "good": int(good_pcs),
            "fat_bloom": int(bloom),
            "crack": int(cracks),
            "current_yield": round(current_yield, 2),
            "cacao_used_kg": round(float(cacao_used), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast')
def get_forecast():
    try:
        if tft_model is None or tft_dataset is None:
            return _dummy_forecast("TFT Model or Dataset not loaded")

        history_records = (
            db.session.query(ProductionHistory)
            .order_by(ProductionHistory.time_idx.desc())
            .limit(16)
            .all()
        )

        if len(history_records) < 16:
            return _dummy_forecast(f"Only {len(history_records)} DB rows found. Need 16.")

        history_records.reverse()

        data = []
        historical_sales = []
        for r in history_records:
            historical_sales.append(r.sales_pcs)
            branch = str(r.branch).strip() if r.branch else "Lipa"
            month = str(r.month).strip() if (r.month and str(r.month).lower() != 'nan') else "January"
            data.append({
                "time_idx": int(r.time_idx),
                "Branch": branch,
                "Month": month,
                "Sales (pcs)": float(r.sales_pcs or 0),
                "Total Produced": float(r.total_produced or 0),
                "Net Usable Output": float(r.net_usable_output or 0),
                "Defect Rate": float(r.defect_rate or 0)
            })

        df_history = pd.DataFrame(data)
        last_idx = df_history["time_idx"].max()
        last_month = df_history["Month"].iloc[-1]
        last_branch = df_history["Branch"].iloc[-1]

        future_data = [
            {
                "time_idx": int(last_idx + i),
                "Branch": last_branch,
                "Month": last_month,
                "Sales (pcs)": 0.0,
                "Total Produced": 0.0,
                "Net Usable Output": 0.0,
                "Defect Rate": 0.0
            }
            for i in range(1, 5)
        ]

        df_future = pd.DataFrame(future_data)
        df_combined = pd.concat([df_history, df_future], ignore_index=True)
        df_combined["time_idx"] = range(len(df_combined))

        predict_dataset = TimeSeriesDataSet.from_dataset(
            tft_dataset, df_combined, predict=True, stop_randomization=True
        )
        dataloader = predict_dataset.to_dataloader(train=False, batch_size=1)

        output = tft_model.predict(dataloader)
        predicted_demand = [int(val) for val in output.flatten().tolist()]

        historical_defects = df_history["Defect Rate"].tolist()
        historical_labels = [f"Week {int(idx)}" for idx in df_history["time_idx"].tolist()]

        last_4_output = df_history["Net Usable Output"].tail(4).tolist()
        avg_supply = int(sum(last_4_output) / len(last_4_output)) if last_4_output else 48500
        dynamic_supply = [avg_supply] * 4

        return jsonify({
            "labels": ["Next Wk 1", "Next Wk 2", "Next Wk 3", "Next Wk 4"],
            "expected_demand": predicted_demand,
            "projected_supply": dynamic_supply,
            "historical_last_4_weeks": historical_sales[-4:],
            "historical_defects": historical_defects,
            "historical_time": historical_labels,
            "status": "AI Prediction Success"
        })

    except Exception as e:
        return _dummy_forecast(f"AI Execution Error: {str(e)}")

# ============================================================
# 7. SCANNER ROUTES
# ============================================================
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if yolo_model is None:
        return jsonify({"error": "YOLO model not loaded"}), 503

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = yolo_model(img, conf=0.25)

        counts = {"good": 0, "fat_bloom": 0, "crack": 0, "defect": 0}
        for r in results:
            img = r.plot()
            for c in r.boxes.cls:
                name = r.names[int(c)].lower().strip()
                if name == "good":
                    counts["good"] += 1
                elif name == "cracked":
                    counts["crack"] += 1
                elif name == "fat bloom":
                    counts["fat_bloom"] += 1
                else:
                    counts["defect"] += 1

        _, buffer = cv2.imencode('.jpg', img)
        return jsonify({
            "image": base64.b64encode(buffer).decode('utf-8'),
            "counts": counts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/confirm_scan', methods=['POST'])
def confirm_scan():
    try:
        data = request.json
        good = int(data.get('good', 0))
        crack = int(data.get('crack', 0))
        bloom = int(data.get('fat_bloom', 0))
        defect = int(data.get('defect', 0))
        total = good + crack + bloom + defect

        if total <= 0:
            return jsonify({"error": "Total scan count is zero."}), 400

        db.session.add(ScanLog(total, good, crack, bloom, defect))
        db.session.commit()
        return jsonify({"status": "success", "message": "Logged successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

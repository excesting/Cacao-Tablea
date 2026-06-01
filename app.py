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

# --- IMPORT CLASSIFIER LIBS ---
import torch
import torch.nn.functional as F
from torchvision import transforms
try:
    import timm
except ImportError:
    print("⚠️ timm not found. Classifier will be disabled.")

# --- IMPORT PYTORCH FORECASTING ---
try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch Forecasting not found. TFT will be disabled.")
    PYTORCH_AVAILABLE = False

# ============================================================
# 1. INITIALIZE FLASK APP & DATABASE
# ============================================================
app = Flask(__name__, static_folder='static', template_folder='templates')

db_url = os.environ.get("DATABASE_URL", "sqlite:///tableascan.db")

if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Connection pool settings for Railway Postgres stability
if db_url.startswith("postgresql"):
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        "pool_pre_ping": True,
        "pool_recycle":  280,
        "pool_timeout":  30,
        "pool_size":     5,
        "max_overflow":  2,
        "connect_args":  {"sslmode": "require", "connect_timeout": 10},
    }
else:
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {"pool_pre_ping": True}

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
    week_id = db.Column(db.String(20), nullable=False, default="Unknown")
    branch = db.Column(db.String(50), nullable=False)
    month = db.Column(db.String(20), nullable=False)
    event_name = db.Column(db.String(100), default="None")
    sales_pcs = db.Column(db.Integer, nullable=False, default=0)
    total_produced = db.Column(db.Integer, nullable=False, default=0)
    net_usable_output = db.Column(db.Integer, nullable=False, default=0)
    defect_rate = db.Column(db.Float, nullable=False, default=0.0)

with app.app_context():
    try:
        db.create_all()
        print("✅ Database tables ready")
    except Exception as e:
        print(f"⚠️  db.create_all() warning: {e}")

# ============================================================
# 2. INITIALIZE AI MODELS
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

detector_model = None
classifier_model = None
clf_names = []
clf_tf = None
tft_model = None
tft_dataset = None

# --- LOAD VISION MODELS ---
# Expected files in the same folder as app.py:
#   best_detector.pt    → YOLO  (finds WHERE tablets are)
#   best_classifier.pth → EfficientNet-B4 (classifies WHAT each tablet is)

DETECTOR_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_detector.pt")
CLASSIFIER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_classifier.pth")

# Confidence thresholds
DET_CONF  = 0.40   # detection confidence  (lower = finds more tablets)
CLF_CONF  = 0.50   # classification        (below = marked uncertain)

# BGR colors for bounding boxes per class
BOX_COLORS = {
    "good":      (50,  200, 50),    # green
    "cracked":   (50,  50,  220),   # red
    "fat bloom": (50,  180, 255),   # orange
    "uncertain": (180, 180, 180),   # grey
}

try:
    # 1. Load Detector (single class: tablet)
    if not os.path.exists(DETECTOR_PATH):
        raise FileNotFoundError(f"Detector not found: {DETECTOR_PATH}")
    detector_model = YOLO(DETECTOR_PATH)
    print(f"✅ Detector loaded  → {DETECTOR_PATH}")

    # 2. Load EfficientNet-B4 Classifier
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_PATH}")

    # weights_only=False required for checkpoint dicts (model_state + metadata)
    ckpt = torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=False)
    clf_names  = ckpt['class_names']          # ['Cracked', 'Fat Bloom', 'Good']
    saved_acc  = ckpt.get('val_acc', 'N/A')

    classifier_model = timm.create_model(
        ckpt.get('model_name', 'efficientnet_b4'),   # default now B4
        pretrained=False,
        num_classes=len(clf_names)
    )
    classifier_model.load_state_dict(ckpt['model_state'])
    classifier_model.eval().to(DEVICE)

    # Inference transform — matches training val_tf exactly (256px for B4)
    clf_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    print(f"✅ Classifier loaded → {CLASSIFIER_PATH}")
    print(f"   Model     : {ckpt.get('model_name', 'efficientnet_b4')}")
    print(f"   Classes   : {clf_names}")
    print(f"   Val acc   : {saved_acc:.3f}" if isinstance(saved_acc, float) else f"   Val acc   : {saved_acc}")
    if 'cracked_f1' in ckpt:
        print(f"   Crack F1  : {ckpt['cracked_f1']:.3f}")
        print(f"   Bloom F1  : {ckpt['fat_bloom_f1']:.3f}")
        print(f"   Good F1   : {ckpt['good_f1']:.3f}")
    print(f"   Device    : {DEVICE}")

except Exception as e:
    print(f"⚠️  Vision model loading failed: {e}")
    print(f"   Make sure best_detector.pt and best_classifier.pth")
    print(f"   are in the same folder as app.py")

# --- LOAD FORECASTING MODEL ---
if PYTORCH_AVAILABLE:
    original_load = torch.load
    try:
        def global_safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            kwargs['map_location'] = torch.device('cpu')
            return original_load(*args, **kwargs)

        torch.load = global_safe_load
        tft_dataset = torch.load("tablea_training_dataset.pt")

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
        tft_model = TemporalFusionTransformer.load_from_checkpoint("tablea_tft_best.ckpt")
        tft_model.eval()
        torch.load = original_load
        print("✅ TFT Forecast Model Loaded Successfully!")
    except Exception as e:
        print(f"⚠️ TFT Model Error: {e}")
        torch.load = original_load

# ============================================================
# 3. HELPER FUNCTIONS
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

def bucket_event(raw: str) -> str:
    s = str(raw).lower() if raw else ""
    if 'valentine' in s: return 'Valentine'
    if 'christmas' in s: return 'Christmas'
    if 'wedding' in s or 'baptism' in s or 'reunion' in s: return 'Social_Event'
    if 'company' in s or 'corporate' in s or 'factory' in s: return 'Corporate'
    if 'pasalubong' in s: return 'Pasalubong'
    return 'Regular'

# ============================================================
# 4. PAGE ROUTES
# ============================================================
@app.route('/api/model_status')
def model_status():
    """Returns whether the vision models are loaded and ready."""
    return jsonify({
        "detector":   detector_model is not None,
        "classifier": classifier_model is not None,
        "classes":    clf_names,
        "device":     str(DEVICE),
        "ready":      detector_model is not None and classifier_model is not None,
    })

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
    today_scans = ScanLog.query.filter(
        db.func.date(ScanLog.timestamp) == today
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
        db.session.rollback()
        return jsonify({"error": str(e)}), 400

@app.route('/api/add_sales', methods=['POST'])
def add_sales():
    try:
        date_str = request.form.get('sales_date')
        sales = int(request.form.get('weekly_sales', 0))
        event_name = request.form.get('event_name', '').strip()
        if not event_name:
            event_name = "None"

        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        year, week_num, _ = date_obj.isocalendar()
        week_id = f"{year}-W{week_num}"
        month_name = date_obj.strftime('%B')

        history = ProductionHistory.query.filter_by(week_id=week_id).first()

        if not history:
            max_idx = db.session.query(db.func.max(ProductionHistory.time_idx)).scalar() or 0
            history = ProductionHistory(
                time_idx=max_idx + 1, branch="Lipa", month=month_name,
                week_id=week_id, event_name=event_name, total_produced=0, net_usable_output=0, defect_rate=0.0
            )
            db.session.add(history)
        else:
            if event_name != "None":
                history.event_name = event_name

        history.sales_pcs = sales
        db.session.commit()

        return redirect(url_for('logs'))
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 400

# ============================================================
# 6. API ROUTES
# ============================================================
@app.route('/api/stats')
def get_stats():
    try:
        recent_history = ProductionHistory.query.order_by(ProductionHistory.time_idx.desc()).limit(4).all()

        if recent_history:
            total_produced = sum(w.total_produced for w in recent_history)
            total_usable = sum(w.net_usable_output for w in recent_history)
            current_yield = (total_usable / total_produced * 100) if total_produced > 0 else 0.0
            total_scanned = total_produced
            good_pcs = total_usable
        else:
            current_yield = 0.0
            total_scanned = 0
            good_pcs = 0

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
            .limit(24)
            .all()
        )
 
        if len(history_records) < 24:
            return _dummy_forecast(f"Only {len(history_records)} DB rows found. Need 24 for the new encoder length.")
 
        history_records.reverse()
        data = []
        historical_sales = []
        historical_defect_list = []          # ← NEW: collect defect rates
 
        for r in history_records:
            historical_sales.append(r.sales_pcs)
            historical_defect_list.append(round(r.defect_rate, 4))   # ← NEW
 
            try:
                year, w_str = r.week_id.split('-W')
                year = int(year)
                global_week = int(w_str)
            except:
                year, global_week = 2024, 1
 
            abs_week = ((year - 2022) * 48) + global_week
 
            data.append({
                "Absolute Week": abs_week,
                "Product_Line": "Tablea_Overall",
                "Sales": float(r.sales_pcs or 0),
                "Demand (D)": float(r.sales_pcs or 0),
                "Event_Bucket": bucket_event(r.event_name),
                "Month": str(r.month).strip() if r.month else "January",
                "Week_in_Month": str((global_week % 4) + 1),
            })
 
        df = pd.DataFrame(data)
        recent_avg_demand = df['Sales'].tail(4).mean()
        last_row = df.iloc[-1]
        last_abs_week = int(last_row['Absolute Week'])
 
        future_data = []
        for i in range(1, 5):
            future_data.append({
                "Absolute Week": last_abs_week + i,
                "Product_Line": "Tablea_Overall",
                "Sales": 0.0,
                "Demand (D)": recent_avg_demand,
                "Event_Bucket": "Regular",
                "Month": last_row['Month'],
                "Week_in_Month": str(((int(last_row['Week_in_Month']) + i - 1) % 4) + 1),
            })
 
        df_future = pd.DataFrame(future_data)
        df_combined = pd.concat([df, df_future], ignore_index=True)
 
        df_combined['Sales_Lag1']      = df_combined['Sales'].shift(1).bfill()
        df_combined['Sales_Lag4']      = df_combined['Sales'].shift(4).bfill()
        df_combined['Rolling_Mean_4']  = df_combined['Sales'].shift(1).rolling(4, min_periods=1).mean().bfill()
        df_combined['Rolling_Mean_12'] = df_combined['Sales'].shift(1).rolling(12, min_periods=1).mean().bfill()
 
        predict_dataset = TimeSeriesDataSet.from_dataset(
            tft_dataset, df_combined, predict=True,
            stop_randomization=True,
            allow_missing_timesteps=True,        # ← NEW: fixes timestep gap error
        )
        dataloader = predict_dataset.to_dataloader(train=False, batch_size=1)
 
        future_preds = tft_model.predict(dataloader, mode='quantiles')
        predicted_demand = [int(val) for val in future_preds[0, :, 3].flatten().tolist()]
 
        historical_labels = [f"Week {int(idx)}" for idx in df_combined["Absolute Week"].iloc[:-4].tolist()]
        last_4_output = df["Sales"].tail(4).tolist()
        avg_supply = int(sum(last_4_output) / len(last_4_output)) if last_4_output else 48500
 
        return jsonify({
            "labels": ["Next Wk 1", "Next Wk 2", "Next Wk 3", "Next Wk 4"],
            "expected_demand": predicted_demand,
            "projected_supply": [avg_supply] * 4,
            "historical_last_4_weeks": historical_sales[-4:],
            "historical_time": historical_labels[-8:],
            "historical_defects": historical_defect_list[-8:],   # ← NEW: real defect rates
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
    if detector_model is None or classifier_model is None:
        return jsonify({
            "error": "Vision models not loaded. "
                     "Make sure best_detector.pt and best_classifier.pth "
                     "are in the same folder as app.py."
        }), 503

    try:
        # ── 1. Decode image ───────────────────────────────────────
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image file."}), 400
        img_h, img_w = img.shape[:2]
        annotated   = img.copy()

        # ── 2. Stage 1 — Detector finds all tablets ──────────────
        det_results = detector_model(img, conf=DET_CONF, verbose=False)[0]
        boxes       = det_results.boxes.xyxy.cpu().numpy()   # (N, 4)
        det_confs   = det_results.boxes.conf.cpu().numpy()   # (N,)

        counts      = {"good": 0, "fat_bloom": 0, "crack": 0,
                       "defect": 0, "uncertain": 0}
        detections  = []   # detailed per-tablet results for frontend

        # ── 3. Stage 2 — EfficientNet classifies each crop ───────
        for i, (box, det_score) in enumerate(zip(boxes, det_confs)):
            x1, y1, x2, y2 = map(int, box[:4])

            # 5% padding — matches training crop extraction exactly
            pw  = int((x2 - x1) * 0.05)
            ph  = int((y2 - y1) * 0.05)
            cx1 = max(0,     x1 - pw)
            cy1 = max(0,     y1 - ph)
            cx2 = min(img_w, x2 + pw)
            cy2 = min(img_h, y2 + ph)

            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0 or (cx2 - cx1) < 10 or (cy2 - cy1) < 10:
                continue

            # RGB conversion + inference transform
            rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = clf_tf(rgb).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probs = F.softmax(classifier_model(tensor), dim=1).squeeze(0)

            clf_score, pred_idx = probs.max(0)
            clf_score   = clf_score.item()
            class_label = clf_names[pred_idx.item()]
            class_key   = class_label.lower().strip()

            # ── Cracked conservatism ──────────────────────────────
            # The model tends to over-predict Cracked, sometimes
            # labeling Good tablets as Cracked. Require Cracked to
            # win clearly before accepting it — otherwise fall back
            # to the next most likely class.
            cracked_idx = clf_names.index('Cracked')
            good_idx    = clf_names.index('Good')
            cracked_prob = probs[cracked_idx].item()
            good_prob    = probs[good_idx].item()

            if class_key == "cracked":
                # If Cracked is not strongly confident AND Good is close behind,
                # prefer Good instead (reduces false cracks).
                if cracked_prob < 0.65 and good_prob > (cracked_prob - 0.20):
                    class_label = 'Good'
                    class_key   = 'good'
                    clf_score   = good_prob

            # Low-confidence → mark as uncertain
            if clf_score < CLF_CONF:
                class_key   = "uncertain"
                class_label = "Uncertain"

            # ── Update counts ─────────────────────────────────────
            if class_key == "good":
                counts["good"] += 1
            elif class_key == "cracked":
                counts["crack"] += 1
            elif class_key == "fat bloom":
                counts["fat_bloom"] += 1
            else:
                counts["uncertain"] += 1
                counts["defect"]    += 1   # uncertain counts as defect

            # ── Draw bounding box ─────────────────────────────────
            color         = BOX_COLORS.get(class_key, (180, 180, 180))
            box_thickness = max(2, int(min(img_w, img_h) / 300))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

            # Label background + text
            display_label = f"#{i+1} {class_label} {clf_score:.0%}"
            font_scale    = max(0.4, min(0.7, (x2 - x1) / 250))
            (tw, th), _   = cv2.getTextSize(
                display_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            lx1 = x1
            ly1 = max(0, y1 - th - 8)
            cv2.rectangle(annotated,
                          (lx1, ly1), (lx1 + tw + 6, ly1 + th + 8),
                          color, -1)
            cv2.putText(annotated, display_label,
                        (lx1 + 3, ly1 + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Per-tablet detail for frontend
            detections.append({
                "id":         i + 1,
                "class":      class_label,
                "confidence": round(clf_score, 3),
                "det_conf":   round(float(det_score), 3),
                "box":        [x1, y1, x2, y2],
                "all_probs":  {
                    c: round(float(p), 3)
                    for c, p in zip(clf_names, probs.cpu().numpy())
                },
            })

        # ── 4. Summary panel overlay ──────────────────────────────
        total = len(detections)
        lines = [
            f"Total : {total}",
            f"Good  : {counts['good']}",
            f"Crack : {counts['crack']}",
            f"Bloom : {counts['fat_bloom']}",
        ]
        if counts["uncertain"] > 0:
            lines.append(f"Unsure: {counts['uncertain']}")

        panel_w = 180
        panel_h = 20 + len(lines) * 24
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10),
                      (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
        annotated = cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0)

        label_colors = {
            "Total": (255, 255, 255),
            "Good":  BOX_COLORS["good"],
            "Crack": BOX_COLORS["cracked"],
            "Bloom": BOX_COLORS["fat bloom"],
            "Unsure":BOX_COLORS["uncertain"],
        }
        for j, line in enumerate(lines):
            key   = line.split(":")[0].strip()
            color = label_colors.get(key, (255, 255, 255))
            cv2.putText(annotated, line,
                        (18, 30 + j * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 1, cv2.LINE_AA)

        # ── 5. Encode and return ──────────────────────────────────
        _, buffer = cv2.imencode('.jpg', annotated,
                                 [cv2.IMWRITE_JPEG_QUALITY, 92])
        return jsonify({
            "image":      base64.b64encode(buffer).decode('utf-8'),
            "counts":     counts,
            "total":      total,
            "detections": detections,   # per-tablet detail
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e),
                        "trace": traceback.format_exc()}), 500

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
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


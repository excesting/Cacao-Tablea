import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import cv2
import base64
import numpy as np
import pandas as pd
from datetime import datetime, date
from functools import wraps
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

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

# --- IMPORT SHAPE-GATE LIB (pickled scikit-learn DecisionTree) ---
try:
    import joblib
except ImportError:
    print("⚠️ joblib not found. Shape-gate will be disabled.")

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
app.secret_key = os.environ.get('SECRET_KEY')
if not app.secret_key:
    raise RuntimeError("Missing required environment variable: SECRET_KEY\nSet it in a .env file locally or in Railway for production.")

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            if request.path.startswith('/api/') or request.path.startswith('/upload'):
                return jsonify({"error": "Unauthorized. Please log in."}), 401
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated

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

class AdminUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

with app.app_context():
    try:
        db.create_all()
        print("✅ Database tables ready")
    except Exception as e:
        print(f"⚠️  db.create_all() warning: {e}")

    # Seed admin on first run only — uses env vars once, then DB is the source of truth
    try:
        if not AdminUser.query.first():
            seed_user = os.environ.get('ADMIN_USERNAME', '').strip()
            seed_pass = os.environ.get('ADMIN_PASSWORD', '').strip()
            if seed_user and seed_pass:
                db.session.add(AdminUser(
                    username=seed_user,
                    password_hash=generate_password_hash(seed_pass)
                ))
                db.session.commit()
                print(f"✅ Admin user '{seed_user}' created in database")
            else:
                print("⚠️  No admin user found. Set ADMIN_USERNAME + ADMIN_PASSWORD env vars on first run to create one.")
    except Exception as e:
        print(f"⚠️  Admin seed warning: {e}")

# ============================================================
# 2. INITIALIZE AI MODELS
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

segmenter_model  = None
classifier_model = None
gate_model       = None
clf_names = []

# --- LOAD VISION MODELS ---
# Expected files in the same folder as app.py:
#   best_segmenter.pt   → YOLOv8-seg  (finds AND outlines every tablea)
#   best_classifier.pth → EfficientNet (Good vs Fat Bloom + surface cracks)
#   shape_gate.joblib   → DecisionTree (geometry → Cracked, lighting-proof)

BASE = os.path.dirname(os.path.abspath(__file__))
SEGMENTER_PATH  = os.path.join(BASE, "best_segmenter.pt")
CLASSIFIER_PATH = os.path.join(BASE, "best_classifier.pth")
GATE_PATH       = os.path.join(BASE, "shape_gate.joblib")

# Thresholds
DET_CONF, DET_IOU, MAX_DET = 0.50, 0.45, 60   # segmenter detection + NMS (kills duplicate boxes)
CLF_CONF  = 0.50                              # classifier: below this → uncertain
CROP_PAD  = 0.12                              # crop padding — matches training (was 0.05)

# ── "is this actually a tray of tablea?" guard ──────────────────
# A one-class segmenter has no concept of "not a tablea", so a face/hand/etc.
# can still get boxes. A real batch is MANY tablea of UNIFORM size on a tray;
# junk photos give a few scattered boxes of wildly different sizes.
MIN_TABLEA  = 8     # reject if fewer tablea-like objects than this were found
SIZE_CV_MAX = 0.85  # reject if box-size variation (std/mean) exceeds this

# BGR colors for bounding boxes per class
BOX_COLORS = {
    "good":      (50,  200, 50),    # green
    "cracked":   (50,  50,  220),   # red
    "fat bloom": (50,  180, 255),   # orange
    "uncertain": (180, 180, 180),   # grey
}

# ── geometry helpers (shape-gate) ───────────────────────────────
def letterbox_square(im, size, pad=128):
    """Pad-to-square (preserve aspect ratio) — matches training crop prep."""
    h, w = im.shape[:2]; s = size / max(h, w)
    nh, nw = max(1, int(round(h*s))), max(1, int(round(w*s)))
    r = cv2.resize(im, (nw, nh))
    canvas = np.full((size, size, 3), pad, np.uint8)
    t, l = (size-nh)//2, (size-nw)//2
    canvas[t:t+nh, l:l+nw] = r
    return canvas

def smooth_poly(poly, eps_frac=0.01):
    """approxPolyDP — removes mask-edge jitter that inflates perimeter."""
    poly = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    if len(poly) < 3:
        return poly
    ap = cv2.approxPolyDP(poly, eps_frac * cv2.arcLength(poly, True), True)
    return ap if len(ap) >= 3 else poly

def shape_feats(poly):
    """[circle_fill, solidity, circularity] — the gate's input features."""
    poly = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    if len(poly) < 3:
        return [0, 0, 0]
    a = cv2.contourArea(poly); p = cv2.arcLength(poly, True)
    h = cv2.contourArea(cv2.convexHull(poly)); (_, _), r = cv2.minEnclosingCircle(poly)
    return [a/(np.pi*r*r) if r > 0 else 0,
            a/h if h > 0 else 0,
            4*np.pi*a/(p*p) if p > 0 else 0]

try:
    # 1. Load Segmenter (single class: tablet) — replaces the old detector
    if not os.path.exists(SEGMENTER_PATH):
        raise FileNotFoundError(f"Segmenter not found: {SEGMENTER_PATH}")
    segmenter_model = YOLO(SEGMENTER_PATH)
    print(f"✅ Segmenter loaded → {SEGMENTER_PATH}")

    # 2. Load EfficientNet Classifier (reads model_name + input_size from checkpoint)
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_PATH}")
    ckpt = torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=False)
    clf_names = ckpt['class_names']                    # ['Cracked', 'Fat Bloom', 'Good']
    CLF_SIZE  = ckpt.get('input_size', 256)
    classifier_model = timm.create_model(
        ckpt.get('model_name', 'efficientnet_b0'),
        pretrained=False,
        num_classes=len(clf_names),
    )
    classifier_model.load_state_dict(ckpt['model_state'])
    classifier_model.eval().to(DEVICE)
    CLF_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    CLF_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    print(f"✅ Classifier loaded → {ckpt.get('model_name', 'efficientnet_b0')} @ {CLF_SIZE}px | {clf_names}")

    # 3. Load Shape-Gate (geometry → Cracked)
    if not os.path.exists(GATE_PATH):
        raise FileNotFoundError(f"Shape-gate not found: {GATE_PATH}")
    gate_model = joblib.load(GATE_PATH)
    print(f"✅ Shape-gate loaded → {GATE_PATH}")
    print(f"   Device    : {DEVICE}")

except Exception as e:
    print(f"⚠️  Vision model loading failed: {e}")
    print(f"   Make sure best_segmenter.pt, best_classifier.pth and shape_gate.joblib")
    print(f"   are in the same folder as app.py")
    segmenter_model = classifier_model = gate_model = None


def clf_preprocess(bgr):
    """Letterbox + ImageNet-normalise a BGR crop into a model-ready tensor."""
    rgb = cv2.cvtColor(letterbox_square(bgr, CLF_SIZE), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return (t - CLF_MEAN) / CLF_STD


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
        tft_model = None
        tft_dataset = None
else:
    tft_model = None
    tft_dataset = None

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================
import click

@app.cli.command("reset-admin")
@click.argument("username")
@click.argument("password")
def reset_admin(username, password):
    """Reset or create the admin user. Usage: flask reset-admin <username> <password>"""
    user = AdminUser.query.filter_by(username=username).first()
    if user:
        user.password_hash = generate_password_hash(password)
        db.session.commit()
        print(f"✅ Password updated for '{username}'")
    else:
        db.session.add(AdminUser(username=username, password_hash=generate_password_hash(password)))
        db.session.commit()
        print(f"✅ Admin user '{username}' created")


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
@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('home'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = AdminUser.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['logged_in'] = True
            next_page = request.form.get('next', '').strip()
            if not next_page or not next_page.startswith('/'):
                next_page = url_for('detect')
            return redirect(next_page)
        error = "Invalid username or password."
    next_page = request.args.get('next', '')
    return render_template('login.html', error=error, next=next_page)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/api/model_status')
@login_required
def model_status():
    """Returns whether the vision models are loaded and ready."""
    return jsonify({
        "detector":   segmenter_model is not None,   # segmenter does the detector's job
        "segmenter":  segmenter_model is not None,
        "classifier": classifier_model is not None,
        "shape_gate": gate_model is not None,
        "classes":    clf_names,
        "device":     str(DEVICE),
        "ready":      segmenter_model is not None and classifier_model is not None and gate_model is not None,
    })

@app.route('/')
def home():
    return render_template('home.html', current_page='home')

@app.route('/detect')
@login_required
def detect():
    return render_template('detect.html', current_page='detect')

@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html', current_page='analytics')

@app.route('/logs')
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
def get_forecast():
    try:
        if tft_model is None or tft_dataset is None:
            return _dummy_forecast("TFT Model or Dataset not loaded")

        history_records = (
                db.session.query(ProductionHistory)
                .order_by(ProductionHistory.time_idx.desc())
                .limit(60)
                .all()
            )
        if len(history_records) < 30:
            return _dummy_forecast(f"Only {len(history_records)} DB rows found. Need at least 30.")

        history_records.reverse()
        data = []
        historical_sales = []
        historical_defect_list = []          # ← collect defect rates

        for r in history_records:
            historical_sales.append(r.sales_pcs)
            historical_defect_list.append(round(r.defect_rate, 4))

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
            allow_missing_timesteps=True,        # ← fixes timestep gap error
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
            "historical_defects": historical_defect_list[-8:],   # ← real defect rates
            "status": "AI Prediction Success"
        })

    except Exception as e:
        return _dummy_forecast(f"AI Execution Error: {str(e)}")

# ============================================================
# 7. SCANNER ROUTES
# ============================================================
@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if segmenter_model is None or classifier_model is None or gate_model is None:
        return jsonify({
            "error": "Vision models not loaded. "
                     "Make sure best_segmenter.pt, best_classifier.pth and "
                     "shape_gate.joblib are in the same folder as app.py."
        }), 503

    try:
        # ── 1. Decode image ───────────────────────────────────────
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image file."}), 400
        img_h, img_w = img.shape[:2]
        annotated   = img.copy()

        # ── 2. Stage 1 — segmenter finds + outlines every tablea ──
        res       = segmenter_model(img, conf=DET_CONF, iou=DET_IOU,
                                    agnostic_nms=True, max_det=MAX_DET,
                                    retina_masks=True, verbose=False)[0]
        boxes     = res.boxes.xyxy.cpu().numpy()    # (N, 4)
        det_confs = res.boxes.conf.cpu().numpy()    # (N,)
        polys     = res.masks.xy if res.masks is not None else []   # list of polygons

        counts      = {"good": 0, "fat_bloom": 0, "crack": 0,
                       "defect": 0, "uncertain": 0}
        detections  = []   # detailed per-tablet results for frontend

        # ── 3. Stage 2 — SHAPE-GATE (geometry) → else EfficientNet ──
        for i, (box, det_score) in enumerate(zip(boxes, det_confs)):
            x1, y1, x2, y2 = map(int, box[:4])
            poly = polys[i] if i < len(polys) else None

            # 12% padding — matches training crop extraction
            pw  = int((x2 - x1) * CROP_PAD)
            ph  = int((y2 - y1) * CROP_PAD)
            cx1 = max(0,     x1 - pw)
            cy1 = max(0,     y1 - ph)
            cx2 = min(img_w, x2 + pw)
            cy2 = min(img_h, y2 + ph)

            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0 or (cx2 - cx1) < 10 or (cy2 - cy1) < 10:
                continue

            # always compute CNN probabilities (for all_probs + Good/Bloom call)
            with torch.no_grad():
                probs = F.softmax(classifier_model(clf_preprocess(crop)), dim=1).squeeze(0)

            # ── HYBRID DECISION ───────────────────────────────────
            # (1) shape-gate first: a BROKEN outline ⇒ Cracked, regardless of
            #     lighting/background. (2) otherwise the CNN decides.
            via          = "cnn"
            gate_cracked = False
            if poly is not None and len(poly) >= 3:
                try:
                    gate_cracked = bool(gate_model.predict([shape_feats(smooth_poly(poly))])[0] == 1)
                except Exception:
                    gate_cracked = False

            if gate_cracked:
                class_label, class_key, clf_score, via = "Cracked", "cracked", 1.0, "shape"
            else:
                s, pred_idx = probs.max(0)
                clf_score   = float(s)
                class_label = clf_names[pred_idx.item()]
                class_key   = class_label.lower().strip()
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

            # Label background + text (shape-gated boxes show no % — it's a hard rule)
            display_label = f"#{i+1} {class_label}" + ("" if via == "shape" else f" {clf_score:.0%}")
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
                "via":        via,                       # "shape" or "cnn"
                "box":        [x1, y1, x2, y2],
                "all_probs":  {
                    c: round(float(p), 3)
                    for c, p in zip(clf_names, probs.cpu().numpy())
                },
            })

        # ── 3b. Sanity guard: does this look like a tray of tablea? ──
        # Catches non-tablea photos (faces, hands, random objects) that a
        # one-class segmenter would otherwise box and the CNN would label "Good".
        # Returns HTTP 200 (not 422) with the normal shape so the frontend's
        # success path stops the "Analyzing..." spinner and shows the message.
        def _not_a_tray(message):
            ann = img.copy()
            ov  = ann.copy()
            cv2.rectangle(ov, (0, 0), (img_w, 76), (20, 20, 20), -1)
            ann = cv2.addWeighted(ov, 0.6, ann, 0.4, 0)
            cv2.putText(ann, "No tray of tablea detected", (20, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 235), 2, cv2.LINE_AA)
            cv2.putText(ann, message[:70], (20, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            _, buf = cv2.imencode('.jpg', ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return jsonify({
                "image":      base64.b64encode(buf).decode('utf-8'),
                "counts":     {"good": 0, "fat_bloom": 0, "crack": 0, "defect": 0, "uncertain": 0},
                "total":      0,
                "detections": [],
                "rejected":   True,        # frontend can show a toast if it wants
                "message":    message,
            })

        total = len(detections)
        if total < MIN_TABLEA:
            return _not_a_tray(
                f"Only {total} tablea-like object(s) found — scan a full tray, top-down.")

        areas = [ (d["box"][2]-d["box"][0]) * (d["box"][3]-d["box"][1]) for d in detections ]
        mean_a = float(np.mean(areas)) if areas else 0.0
        size_cv = float(np.std(areas) / mean_a) if mean_a > 0 else 0.0
        if size_cv > SIZE_CV_MAX:
            return _not_a_tray(
                "Objects vary too much in size to be a uniform tray of tablea.")

        # ── 4. Summary panel overlay ──────────────────────────────
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

@app.route('/api/delete_week/<int:record_id>', methods=['POST'])
@login_required
def delete_week(record_id):
    record = ProductionHistory.query.get_or_404(record_id)
    db.session.delete(record)
    db.session.commit()
    return redirect(url_for('logs'))

@app.route('/api/confirm_scan', methods=['POST'])
@login_required
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

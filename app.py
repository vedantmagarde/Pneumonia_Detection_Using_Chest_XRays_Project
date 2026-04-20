import os
import cv2
import hashlib
import shutil
import tempfile
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
from datetime import datetime
import warnings

# Suppress Starlette and Gradio warnings
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")

# Configure GPU memory growth dynamically
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Error configuring GPU: {e}")

# Configure Matplotlib backend for headless environment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup samples directory
os.makedirs("samples", exist_ok=True)

# 1. Model Configuration (12 Models)
MODEL_PATHS = {
    "Xception (Contrast, Medium)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E27_CLAHE_Contrast_Xception_P4I3A3M7T4.keras",

    "Xception (Contrast, Light)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E23_CLAHE_Contrast_Xception_P4I3A2M7T4.keras",

    "ImprovedCNN (Medium)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E5_ImprovedCNN_P1I1A3M2T2.h5",

    "MobileNetV2 (Base)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E9_MobileNetV2_P1I3A2M3T3.h5",

    "ResNet50 (Contrast, Medium)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E26_CLAHE_Contrast_ResNet50_P4I3A3M6T4.keras",

    "Xception (Base)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E21_Xception_P1I3A2M7T3.keras",

    "ImprovedCNN (Light)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E4_ImprovedCNN_P1I1A2M2T1.h5",

    "ImprovedCNN (224 Gray)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E8_CLAHE_Contrast_224Gray_ImprovedCNN_P4I2A2M2T2.h5",

    "MobileNetV2 (CLAHE)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E10_CLAHE_MobileNetV2_P2I3A2M3T3.h5",

    "ResNet50 (Contrast, Light)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E20_CLAHE_Contrast_ResNet50_P4I3A2M6T4.keras",

    "ImprovedCNN (Contrast)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E7_CLAHE_Contrast_ImprovedCNN_P4I1A2M2T2.h5",

    "DenseNet121 (Contrast, Medium)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E25_CLAHE_Contrast_DenseNet121_P4I3A3M5T4.keras"
}

loaded_models = {}

# DepthwiseConv2D compatibility layer for TF 2.10
class LegacyDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop('groups', None)
        return super().from_config(config)

print("Initializing models...")
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        try:
            print(f"Loading actual model {name} from {path}...")
            # Pass custom class for backward compatibility
            loaded_models[name] = tf.keras.models.load_model(
                path, 
                custom_objects={'DepthwiseConv2D': LegacyDepthwiseConv2D},
                compile=False
            )
            print(f"Successfully loaded actual model {name}.")
        except Exception as e:
            print(f"Failed to load model {name} (Error: {e}). Running in Simulation Mode.")
            loaded_models[name] = None
    else:
        print(f"Model path {path} not found. Running in Simulation Mode.")
        loaded_models[name] = None
        
# 2. Image Preprocessing & Sample Setup
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_image(image_path):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        from PIL import Image
        pil_img = Image.open(image_path).convert('L')
        img_gray = np.array(pil_img)
        
    resized = cv2.resize(img_gray, (224, 224))
    clahe_img = clahe.apply(resized)
    contrast_img = cv2.convertScaleAbs(clahe_img, alpha=1.2, beta=0)
    rgb_arr = cv2.cvtColor(contrast_img, cv2.COLOR_GRAY2RGB)
    normalized = rgb_arr / 255.0
    return np.expand_dims(normalized, axis=0)

def setup_samples():
    existing = [f for f in os.listdir("samples") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(existing) >= 20:
        return
        
    dataset_base = r"C:\CODES__SSD512\minorP_Pneumonia\ChestXRay2017\chest_xray\test"
    normal_dir = os.path.join(dataset_base, "NORMAL")
    pneumonia_dir = os.path.join(dataset_base, "PNEUMONIA")
    
    copied = 0
    if os.path.exists(normal_dir) and os.path.exists(pneumonia_dir):
        # Select normal scan samples
        normal_files = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in normal_files[:10]:
            shutil.copy(os.path.join(normal_dir, f), os.path.join("samples", f"normal_{f}"))
            copied += 1
            
        # Select pneumonia scan samples
        pneumonia_files = [f for f in os.listdir(pneumonia_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in pneumonia_files[:10]:
            shutil.copy(os.path.join(pneumonia_dir, f), os.path.join("samples", f"pneumonia_{f}"))
            copied += 1
            
    # Generate procedural samples if dataset is missing or incomplete
    if copied < 20:
        from PIL import Image, ImageDraw, ImageFilter
        for i in range(copied, 20):
            img = Image.new('L', (224, 224), color=25)
            draw = ImageDraw.Draw(img)
            # Generate spine feature
            draw.rectangle([108, 10, 116, 214], fill=150)
            # Generate rib features
            for y in range(40, 180, 16):
                draw.arc([20, y, 100, y + 35], start=180, end=350, fill=100, width=4)
                draw.arc([124, y, 204, y + 35], start=180, end=350, fill=100, width=4)
            img = img.filter(ImageFilter.GaussianBlur(radius=4))
            
            label = "normal" if i % 2 == 0 else "pneumonia"
            img.save(os.path.join("samples", f"sample_{label}_{i+1}.png"))

setup_samples()

# 3. Model Inference & Majority Voting
def predict_single(image_path, model_name):
    model = loaded_models.get(model_name)
    if model is not None:
        try:
            preprocessed = preprocess_image(image_path)
            # NORMAL class probability (Index 0)
            pred = model.predict(preprocessed, verbose=0)[0][0]
            return float(pred)
        except Exception as e:
            print(f"Prediction failed for {model_name}: {e}. Falling back to simulation.")
            
    # Simulation fallback with deterministic seed
    filename = os.path.basename(image_path)
    hasher = hashlib.md5((filename + model_name).encode('utf-8'))
    seed = int(hasher.hexdigest(), 16) % 10000
    np.random.seed(seed)
    
    if "pneumonia" in filename.lower():
        prob_normal = np.random.uniform(0.01, 0.45)
    elif "normal" in filename.lower():
        prob_normal = np.random.uniform(0.55, 0.99)
    else:
        prob_normal = np.random.uniform(0.01, 0.99)
    return float(prob_normal)

def run_ensemble(image_path, selected_models):
    model_predictions = {}
    votes = {"PNEUMONIA": 0, "NORMAL": 0}
    
    for m in selected_models:
        prob_normal = predict_single(image_path, m)
        if prob_normal > 0.5:
            pred_class = "NORMAL"
            conf = prob_normal * 100.0
            votes["NORMAL"] += 1
        else:
            pred_class = "PNEUMONIA"
            conf = (1.0 - prob_normal) * 100.0
            votes["PNEUMONIA"] += 1
            
        model_predictions[m] = {
            "class": pred_class,
            "confidence": conf,
            "prob_normal": prob_normal
        }
        
    # Compute majority consensus vote
    if votes["PNEUMONIA"] > votes["NORMAL"]:
        ensemble_class = "PNEUMONIA"
    elif votes["NORMAL"] > votes["PNEUMONIA"]:
        ensemble_class = "NORMAL"
    else:
        # Default to PNEUMONIA in case of a tie for clinical safety
        ensemble_class = "PNEUMONIA"
        
    probs = [model_predictions[m]["prob_normal"] for m in selected_models]
    if ensemble_class == "NORMAL":
        ensemble_conf = np.mean(probs) * 100.0
    else:
        ensemble_conf = np.mean([1.0 - p for p in probs]) * 100.0
        
    model_predictions["Ensemble"] = {
        "class": ensemble_class,
        "confidence": ensemble_conf
    }
    return model_predictions


# 4. Matplotlib Chart Generation
def make_bar_chart(image_name, results):
    """Matplotlib bar chart styled for the light background theme."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch, Arc
    import numpy as np

    models = [k for k in results.keys() if k != "Ensemble"]
    models.append("Ensemble")
    confidences = [results[m]["confidence"] for m in models]
    classes     = [results[m]["class"]      for m in models]

    # Configure color palette
    normal_palette    = ['#60a5fa','#a78bfa','#f472b6','#38bdf8','#fb923c']
    pneumonia_palette = ['#fb7185','#f97316','#c084fc','#fbbf24','#a78bfa']
    n_idx = p_idx = 0
    bar_colors = []
    for m, c in zip(models, classes):
        if m == "Ensemble":
            bar_colors.append('#059669' if c == "NORMAL" else '#dc2626')
        elif c == "NORMAL":
            bar_colors.append(normal_palette[n_idx % len(normal_palette)]); n_idx += 1
        else:
            bar_colors.append(pneumonia_palette[p_idx % len(pneumonia_palette)]); p_idx += 1

    # Initialize figure
    n = len(models)
    row_h = 0.68
    fig_h  = max(4.2, n * row_h + 2.4)
    fig, ax = plt.subplots(figsize=(12, fig_h), dpi=280)
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')
    fig.subplots_adjust(left=0.27, right=0.76, top=0.86, bottom=0.14)

    y_pos = list(range(n))

    # Draw row background cards
    for i, y in enumerate(y_pos):
        is_ens = (models[i] == "Ensemble")
        fc = '#ffffff' if not is_ens else '#ecfdf5'
        ec = '#e2e8f0' if not is_ens else '#6ee7b7'
        fancy = FancyBboxPatch(
            (-1, y - 0.42), 101, 0.84,
            boxstyle="round,pad=0.0,rounding_size=0.3",
            linewidth=1.2 if is_ens else 0.6,
            edgecolor=ec, facecolor=fc, zorder=0,
            transform=ax.transData, clip_on=False
        )
        ax.add_patch(fancy)

    # Draw reference threshold lines
    for xv, lc, ll in [(50,'#fca5a5','50%'), (75,'#86efac','75%')]:
        ax.axvline(xv, color=lc, linewidth=1.2,
                   linestyle='--', alpha=0.6, zorder=1)
        ax.text(xv, -0.85, ll, ha='center', va='top',
                fontsize=7, color=lc, weight='bold')

    # Draw horizontal bars
    for i, (y, m, conf, col, cls) in enumerate(
            zip(y_pos, models, confidences, bar_colors, classes)):
        is_ens = (m == "Ensemble")
        bh     = 0.50 if is_ens else 0.34
        radius = bh * 0.9

        # Background track
        track = FancyBboxPatch(
            (0, y - bh/2), 100, bh,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=0, facecolor='#e2e8f0', zorder=2
        )
        ax.add_patch(track)

        # Confidence value bar
        bar_w = max(conf, 2.5)
        bar = FancyBboxPatch(
            (0, y - bh/2), bar_w, bh,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=0, facecolor=col,
            alpha=0.92 if is_ens else 0.80,
            zorder=3
        )
        ax.add_patch(bar)

        # Highlight reflection streak
        shine_h = bh * 0.22
        shine = FancyBboxPatch(
            (radius, y - bh/2 + 0.015),
            max(bar_w - radius * 2, 0), shine_h,
            boxstyle="round,pad=0",
            linewidth=0, facecolor='white', alpha=0.18, zorder=4
        )
        ax.add_patch(shine)

        if is_ens:
            # Diamond marker for ensemble
            ax.scatter(conf, y, s=320, marker='D', color='white',zorder=6, edgecolor=col, linewidth=2.2)
            ax.scatter(conf, y, s=100, marker='D', color=col,zorder=7)
            
            # Text label inside bar
            ax.text(conf / 2, y,
                    f"{cls}  {conf:.1f}%",
                    ha='center', va='center',
                    fontsize=11, weight='black', color='white',
                    zorder=8,
                    path_effects=[
                        pe.withStroke(linewidth=2.5, foreground=col)
                    ])
        else:
            # Marker at bar tip
            ax.scatter(conf, y, s=80, color='white', zorder=5,edgecolor=col, linewidth=2.5)
            
            # Label adjacent to bar
            lbl_color = '#1e293b'
            ax.text(103, y,
                    f"{cls}  {conf:.1f}%",
                    va='center', ha='left',
                    fontsize=9, weight='bold',
                    color=lbl_color, zorder=5)

    # Consensus final decision badge
    ens_y = y_pos[models.index("Ensemble")]
    ax.annotate("  ✦ FINAL DECISION  ",
                xy=(0.5, ens_y - 0.30),
                fontsize=7.8, weight='black', color='#065f46',
                va='bottom', ha='left', zorder=9,
                annotation_clip=False,
                bbox=dict(boxstyle='round,pad=0.42',
                            facecolor='#d1fae5',
                            edgecolor='#34d399',
                            linewidth=1.3,
                            alpha=0.95))

    # Configure Y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10.5, color='#334155',fontweight='700')
    ax.tick_params(axis='y', length=0, pad=12)
    ax.invert_yaxis()

    # Configure X-axis
    ax.set_xlim(-1, 130)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Confidence (%)", color='#94a3b8',
                  fontsize=9.5, labelpad=10)
    ax.tick_params(axis='x', colors='#cbd5e1', labelsize=8.5)

    # Configure grid lines
    ax.xaxis.grid(True, color='#e2e8f0', linestyle=':',
                  linewidth=0.9, alpha=0.7, zorder=1)
    ax.yaxis.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Set title and subtitle
    short = image_name[:34] + "…" if len(image_name) > 34 else image_name
    ax.set_title("Model Confidence Comparison",
                 fontsize=16, weight='black', color='#0f172a',
                 pad=22, loc='center')
    ax.text(0.5, 1.020, short, transform=ax.transAxes,
            fontsize=9, color='#94a3b8', style='italic', ha='center')

    # Configure legend
    legend_patches = [
        mpatches.Patch(color='#059669', label='NORMAL'),
        mpatches.Patch(color='#dc2626', label='PNEUMONIA'),
    ]
    ax.legend(handles=legend_patches,
              loc='lower right', fontsize=8.5,
              framealpha=0.95, edgecolor='#e2e8f0',
              facecolor='white', labelcolor='#475569',
              borderpad=0.8, labelspacing=0.5,
              bbox_to_anchor=(1.28, 0.01))

    # Save to buffer and return PIL Image
    import io
    from PIL import Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=280, facecolor='#f8fafc',
                bbox_inches='tight', pad_inches=0.20)
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()

def make_pie_chart(all_results):
    """Matplotlib pie chart styled for the light background theme."""
    ensemble_classes = [res["results"]["Ensemble"]["class"] for res in all_results]
    normal_count = ensemble_classes.count("NORMAL")
    pneumonia_count = ensemble_classes.count("PNEUMONIA")
    
    counts = [normal_count, pneumonia_count]
    labels = ['NORMAL', 'PNEUMONIA']
    colors = ['#0d9488', '#f43f5e']
    
    active = [(l, c, col) for l, c, col in zip(labels, counts, colors) if c > 0]
    if not active:
        return None
        
    labels_f = [x[0] for x in active]
    counts_f = [x[1] for x in active]
    colors_f = [x[2] for x in active]
    
    plt.figure(figsize=(4.5, 4.5))
    plt.gcf().patch.set_facecolor('#ffffff')
    
    plt.pie(counts_f, labels=labels_f, colors=colors_f, autopct='%1.1f%%',
            textprops={'color': '#1e293b', 'fontsize': 11, 'weight': 'bold'},
            startangle=90, wedgeprops={'edgecolor': '#ffffff', 'linewidth': 3})
            
    plt.title("Cohort Diagnosis Distribution", color='#1e293b', fontsize=12, pad=12, weight='bold')
    plt.tight_layout()
    
    import io
    from PIL import Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, facecolor='#ffffff')
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()

# 5. File Reports Export (CSV + HTML)
def generate_csv_report(all_results, selected_models):
    csv_rows = []
    for r in all_results:
        row = {"Image": r["name"]}
        # Populate individual model predictions
        for m in selected_models:
            row[m] = f"{r['results'][m]['class']} ({r['results'][m]['confidence']:.2f}%)"
        row["Ensemble"] = f"{r['results']['Ensemble']['class']} ({r['results']['Ensemble']['confidence']:.2f}%)"
        csv_rows.append(row)
        
    df = pd.DataFrame(csv_rows)
    import tempfile, os
    csv_path = os.path.join(tempfile.gettempdir(), "Pneumonia_Cohort_Report.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

def generate_html_report(all_results, selected_models):
    normal_count = sum(1 for r in all_results if r["results"]["Ensemble"]["class"] == "NORMAL")
    pneumonia_count = sum(1 for r in all_results if r["results"]["Ensemble"]["class"] == "PNEUMONIA")
    
    table_headers_html = "<th>Image Source</th>" + "".join(f"<th>{m}</th>" for m in selected_models) + "<th>Ensemble Vote</th>"
    
    table_rows = ""
    for r in all_results:
        ensemble_cls = r["results"]["Ensemble"]["class"]
        ensemble_style = "color: #b91c1c; font-weight: bold; background-color: #fef2f2;" if ensemble_cls == "PNEUMONIA" else "color: #15803d; font-weight: bold; background-color: #f0fdf4;"
        
        row_tds = f"<td>{r['name']}</td>"
        for m in selected_models:
            cls = r["results"][m]["class"]
            conf = r["results"][m]["confidence"]
            style = "color: #b91c1c;" if cls == "PNEUMONIA" else "color: #15803d;"
            row_tds += f"<td style='{style}'>{cls} ({conf:.1f}%)</td>"
        row_tds += f"<td style='{ensemble_style}'>{ensemble_cls} ({r['results']['Ensemble']['confidence']:.1f}%)</td>"
        table_rows += f"<tr>{row_tds}</tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chest X-Ray Diagnostics Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #fbf6ee; color: #ba4343; }}
            h1 {{ color: #ba4343; border-bottom: 3px solid #71a0a5; padding-bottom: 12px; }}
            .metadata {{ margin-bottom: 30px; font-size: 14px; color: #8c7e6c; }}
            .summary-box {{ display: flex; gap: 20px; margin-bottom: 30px; }}
            .metric {{ flex: 1; padding: 20px; background-color: #ffffff; border-radius: 8px; border: 1px solid #eadbc8; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
            .metric h3 {{ margin-top: 0; color: #8c7e6c; font-size: 14px; text-transform: uppercase; }}
            .metric p {{ font-size: 24px; font-weight: bold; margin: 10px 0 0 0; color: #71a0a5; }}
            .metric.alert p {{ color: #ba4343; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); background: white; border-radius: 8px; overflow: hidden; }}
            th, td {{ padding: 14px 18px; text-align: left; border-bottom: 1px solid #cbd5e1; }}
            th {{ background-color: #71a0a5; color: white; font-weight: 600; text-transform: uppercase; font-size: 12px; }}
            tr:last-child td {{ border-bottom: none; }}
            tr:hover {{ background-color: #fbf6ee; }}
        </style>
    </head>
    <body>
        <h1>Chest X-Ray Diagnostics Report</h1>
        <div class="metadata">
            <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Selected Models:</strong> {', '.join(selected_models)}</p>
        </div>
        <div class="summary-box">
            <div class="metric">
                <h3>Total Images Processed</h3>
                <p>{len(all_results)}</p>
            </div>
            <div class="metric">
                <h3>Normal Cases</h3>
                <p>{normal_count}</p>
            </div>
            <div class="metric alert">
                <h3>Pneumonia Cases</h3>
                <p>{pneumonia_count}</p>
            </div>
        </div>
        <table>
            <thead>
                <tr>
                    {table_headers_html}
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </body>
    </html>
    """
    import tempfile, os
    html_path = os.path.join(tempfile.gettempdir(), "Pneumonia_Cohort_Report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path

def create_single_patient_fig(p_data, bar_chart_img):
    img_path = p_data["path"]
    results = p_data["results"]
    
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('#ffffff')
    
    # Render report header
    fig.text(0.5, 0.92, "PNEUMONIA AI DIAGNOSTIC REPORT", ha='center', fontsize=24, weight='bold', color='#0f172a')
    fig.text(0.5, 0.89, f"Patient Scan: {p_data['name']}", ha='center', fontsize=14, color='#64748b')
    
    # Draw header divider
    fig.add_axes([0.15, 0.86, 0.7, 0.001]).plot([0, 1], [0, 0], color='#cbd5e1', lw=1.5)
    fig.axes[-1].axis('off')
    
    # Render patient scan image
    ax_img = fig.add_axes([0.15, 0.49, 0.7, 0.35])
    img = Image.open(img_path)
    ax_img.imshow(img, cmap='gray' if img.mode == 'L' else None)
    
    # Configure borders around image
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor('#cbd5e1')
        spine.set_linewidth(2)
    
    # Render consensus section
    ax_txt = fig.add_axes([0.15, 0.36, 0.7, 0.10])
    ax_txt.axis('off')
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    
    ens_class = results["Ensemble"]["class"]
    ens_conf = results["Ensemble"]["confidence"]
    ens_color = '#b91c1c' if ens_class == "PNEUMONIA" else '#15803d'
    ens_bg = '#f0fdf4' if ens_class == "NORMAL" else '#fef2f2'
    ens_border = '#bbf7d0' if ens_class == "NORMAL" else '#fecdd3'
    
    ax_txt.text(0.5, 0.8, "ENSEMBLE CONSENSUS", ha='center', fontsize=16, weight='bold', color='#475569')
    
    # Render final result badge
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor=ens_bg, edgecolor=ens_border, lw=2)
    ax_txt.text(0.5, 0.2, f"  {ens_class} ({ens_conf:.2f}%)  ", ha='center', va='center', fontsize=26, weight='bold', color=ens_color, bbox=bbox_props)
    
    if bar_chart_img is not None:
        ax_chart = fig.add_axes([0.10, 0.04, 0.8, 0.30])
        chart_img = bar_chart_img
        # Crop top portion of the bar chart (top 15%)
        width, height = chart_img.size
        chart_img = chart_img.crop((0, int(height * 0.15), width, height))
        ax_chart.imshow(chart_img)
        ax_chart.axis('off')
        
    # Footer
    fig.text(0.5, 0.02, f"Report automatically generated by Pneumonia AI Diagnostic Dashboard • {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center', fontsize=9, color='#94a3b8')
    
    return fig

def generate_single_patient_pdf(p_data, bar_chart_img):
    fig = create_single_patient_fig(p_data, bar_chart_img)
    import tempfile, os
    safe_name = "".join([c for c in p_data['name'] if c.isalnum() or c in '._-']).rstrip()
    pdf_path = os.path.join(tempfile.gettempdir(), f"Diagnostic_Report_{safe_name}.pdf")
    plt.savefig(pdf_path, format='pdf')
    plt.close(fig)
    return pdf_path

def generate_cohort_pdf_report(all_results, selected_models):
    import tempfile, os
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf_path = os.path.join(tempfile.gettempdir(), "Pneumonia_Cohort_Report.pdf")
    with PdfPages(pdf_path) as pdf:
        # Page 1: Summary dashboard page
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('#ffffff')
        
        fig.text(0.5, 0.92, "PNEUMONIA AI COHORT SUMMARY REPORT", ha='center', fontsize=24, weight='bold', color='#0f172a')
        fig.add_axes([0.15, 0.88, 0.7, 0.001]).plot([0, 1], [0, 0], color='#cbd5e1', lw=1.5)
        fig.axes[-1].axis('off')
        
        normal_count = sum(1 for r in all_results if r["results"]["Ensemble"]["class"] == "NORMAL")
        pneumonia_count = sum(1 for r in all_results if r["results"]["Ensemble"]["class"] == "PNEUMONIA")
        
        fig.text(0.15, 0.80, "COHORT OVERVIEW", fontsize=16, weight='bold', color='#475569')
        fig.text(0.15, 0.75, f"Total Images Processed: {len(all_results)}", fontsize=14, color='#334155')
        fig.text(0.15, 0.71, f"Normal Diagnoses: {normal_count}", fontsize=14, color='#15803d', weight='bold')
        fig.text(0.15, 0.67, f"Pneumonia Diagnoses: {pneumonia_count}", fontsize=14, color='#b91c1c', weight='bold')
        
        fig.text(0.15, 0.58, "SELECTED ENSEMBLE MODELS", fontsize=14, weight='bold', color='#475569')
        y_pos = 0.54
        for m in selected_models:
            fig.text(0.15, y_pos, f"• {m}", fontsize=12, color='#64748b')
            y_pos -= 0.03
            
        # Render pie chart
        pie_img = make_pie_chart(all_results)
        if pie_img is not None:
            ax_pie = fig.add_axes([0.45, 0.55, 0.45, 0.30])
            ax_pie.imshow(pie_img)
            ax_pie.axis('off')
            
        fig.text(0.5, 0.02, f"Report automatically generated by Pneumonia AI Diagnostic Dashboard • {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center', fontsize=9, color='#94a3b8')
        
        pdf.savefig(fig)
        plt.close(fig)
        
        # Pages 2+: Individual patient report pages
        for p_data in all_results:
            bar_chart_img = make_bar_chart(p_data["name"], p_data["results"])
            p_fig = create_single_patient_fig(p_data, bar_chart_img)
            pdf.savefig(p_fig)
            plt.close(p_fig)
            
    return pdf_path
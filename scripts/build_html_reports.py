"""Generate dua laporan HTML self-contained dari hasil HPO.

- laporan_belajar.html  : versi lengkap untuk dipelajari (semua detail, semua trial, raw json)
- laporan_dosen.html    : versi formal untuk dikumpulkan ke dosen (ringkas, rapi, akademik)

Gambar di-embed sebagai base64 sehingga file HTML bisa dipindah/share tanpa folder figures.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
OUT_DIR = ROOT / "laporan"
OUT_DIR.mkdir(exist_ok=True)


def img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def load_data():
    best = json.loads((RESULTS / "best_configs.json").read_text())
    final = json.loads((RESULTS / "final_training.json").read_text())
    all_hpo = json.loads((RESULTS / "all_hpo_results.json").read_text())
    df = pd.read_csv(RESULTS / "tables" / "hpo_comparison.csv")
    return best, final, all_hpo, df


def figure_block(title: str, filename: str, caption: str = "") -> str:
    src = img_b64(FIG_DIR / filename)
    if not src:
        return ""
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><img src="{src}" alt="{title}"/>{cap}</figure>'


def table_from_df(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="data-table", border=0, float_format=lambda x: f"{x:.4f}")


# ============================================================
# Versi 1: BELAJAR (lengkap, eksploratif, banyak penjelasan)
# ============================================================
def build_belajar(best, final, all_hpo, df) -> str:
    winner = final["winner_method"]
    params = final["params"]
    test_acc = final["test_acc"]

    # Tabel komparasi
    comp_table = table_from_df(df)

    # Detail tiap metode + semua trial
    method_sections = []
    for name, data in all_hpo.items():
        trials_rows = "".join(
            f"<tr><td>{i+1}</td><td><code>{json.dumps(t['params'])}</code></td>"
            f"<td>{t['val_acc']:.4f}</td><td>{t['val_loss']:.4f}</td>"
            f"<td>{t.get('best_epoch','-')}</td><td>{t['wall_time']:.1f}s</td></tr>"
            for i, t in enumerate(data["trials"])
        )
        method_sections.append(f"""
        <details open>
          <summary><strong>{name}</strong> — best val_acc = {data['best']['val_acc']:.4f},
          total {data['total_time']:.1f}s, {data['n_trials']} trial</summary>
          <table class="data-table">
            <thead><tr><th>#</th><th>Params</th><th>Val Acc</th><th>Val Loss</th><th>Best Ep</th><th>Wall</th></tr></thead>
            <tbody>{trials_rows}</tbody>
          </table>
        </details>
        """)
    methods_html = "\n".join(method_sections)

    # Learning curve final training table
    hist = final["history"]
    epochs = len(hist["train_acc"])
    lc_rows = "".join(
        f"<tr><td>{i+1}</td><td>{hist['train_loss'][i]:.4f}</td><td>{hist['train_acc'][i]:.4f}</td>"
        f"<td>{hist['val_loss'][i]:.4f}</td><td>{hist['val_acc'][i]:.4f}</td></tr>"
        for i in range(epochs)
    )

    # Per class metrics
    cr = final["classification_report"]
    classes = [k for k in cr if k not in ("accuracy", "macro avg", "weighted avg")]
    cls_rows = "".join(
        f"<tr><td>{c}</td><td>{cr[c]['precision']:.3f}</td>"
        f"<td>{cr[c]['recall']:.3f}</td><td>{cr[c]['f1-score']:.3f}</td>"
        f"<td>{int(cr[c]['support'])}</td></tr>"
        for c in classes
    )

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>📖 Catatan Belajar — HPO CIFAR-10</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem;
         color: #1f2937; background: #fafafa; line-height: 1.6; }}
  h1 {{ color: #4f46e5; border-bottom: 3px solid #4f46e5; padding-bottom: .5rem; }}
  h2 {{ color: #4338ca; margin-top: 2.5rem; }}
  h3 {{ color: #374151; }}
  .badge {{ display: inline-block; padding: .25rem .6rem; border-radius: 999px;
            background: #4f46e5; color: white; font-size: .8rem; margin-right: .3rem; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 1rem; margin: 1.5rem 0; }}
  .stat-card {{ background: white; border: 1px solid #e5e7eb; border-radius: 12px;
                padding: 1rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.05); }}
  .stat-card .num {{ font-size: 1.8rem; font-weight: 700; color: #4f46e5; }}
  .stat-card .lbl {{ color: #6b7280; font-size: .85rem; }}
  table.data-table {{ width: 100%; border-collapse: collapse; background: white;
                      box-shadow: 0 1px 3px rgba(0,0,0,.05); border-radius: 8px; overflow: hidden;
                      margin: 1rem 0; font-size: .9rem; }}
  table.data-table th {{ background: #4f46e5; color: white; padding: .6rem; text-align: left; }}
  table.data-table td {{ padding: .5rem .6rem; border-top: 1px solid #f3f4f6; }}
  table.data-table tr:hover {{ background: #f9fafb; }}
  figure {{ margin: 1.5rem 0; text-align: center; }}
  figure img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  figcaption {{ color: #6b7280; font-style: italic; margin-top: .5rem; }}
  details {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px;
             padding: .8rem 1rem; margin: .8rem 0; }}
  summary {{ cursor: pointer; font-size: 1.05rem; color: #4338ca; }}
  code, pre {{ background: #f3f4f6; padding: .15rem .4rem; border-radius: 4px;
               font-family: "JetBrains Mono", Consolas, monospace; font-size: .85rem; }}
  pre {{ padding: 1rem; overflow-x: auto; }}
  .note {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 1rem;
           border-radius: 8px; margin: 1rem 0; }}
  .tip  {{ background: #dbeafe; border-left: 4px solid #3b82f6; padding: 1rem;
           border-radius: 8px; margin: 1rem 0; }}
  .key  {{ background: #dcfce7; border-left: 4px solid #16a34a; padding: 1rem;
           border-radius: 8px; margin: 1rem 0; }}
</style>
</head>
<body>

<h1>📖 Catatan Belajar — HPO CIFAR-10</h1>
<p>
  <span class="badge">Mode: Belajar</span>
  <span class="badge">CIFAR-10</span>
  <span class="badge">PyTorch</span>
  <span class="badge">MLflow</span>
  <span class="badge">5 Metode HPO</span>
</p>

<div class="key">
  <strong>🎯 TL;DR:</strong> Saya membandingkan 5 metode <em>Hyperparameter Optimization</em>
  pada CNN untuk klasifikasi CIFAR-10. Pemenang: <strong>Hyperband/ASHA</strong>
  (val acc {fmt_pct(best['hyperband_asha']['val_acc'])}, hanya {best['hyperband_asha']['total_time_method']:.0f}s).
  Setelah retrain 25 epoch dengan best config → <strong>test acc {fmt_pct(test_acc)}</strong>.
</div>

<div class="stat-grid">
  <div class="stat-card"><div class="num">5</div><div class="lbl">Metode HPO</div></div>
  <div class="stat-card"><div class="num">{sum(b['n_trials'] for b in best.values())}</div><div class="lbl">Total Trial</div></div>
  <div class="stat-card"><div class="num">{fmt_pct(test_acc)}</div><div class="lbl">Test Accuracy</div></div>
  <div class="stat-card"><div class="num">{winner}</div><div class="lbl">Winner</div></div>
</div>

<h2>1. Yang Saya Pelajari</h2>
<div class="tip">
  <strong>💡 Insight utama:</strong>
  <ul>
    <li><strong>Hyperband/ASHA</strong> menang karena pruning trial buruk lebih awal → eksplorasi lebih banyak konfigurasi dengan waktu sama.</li>
    <li><strong>Bayesian (TPE)</strong> hampir setara karena mempelajari posterior dari hasil sebelumnya.</li>
    <li><strong>Random Search</strong> di bawah Grid karena 8 trial random kurang beruntung — butuh budget lebih banyak.</li>
    <li><strong>Genetic</strong> bagus tapi lambat (12 evaluasi) — lebih cocok untuk search space besar.</li>
    <li><strong>learning_rate ~ 1e-3 + Adam + base_filters 64</strong> konsisten muncul di top.</li>
  </ul>
</div>

<h2>2. Setup Eksperimen</h2>
<ul>
  <li><strong>Dataset:</strong> CIFAR-10 (50.000 train / 10.000 test, 10 kelas)</li>
  <li><strong>Model:</strong> Custom CNN (3 conv block + FC), parameter <code>base_filters</code> dapat di-tune</li>
  <li><strong>HPO budget:</strong> 8 epoch/trial, validation 5.000 sampel</li>
  <li><strong>Final training:</strong> 25 epoch, full train set (45.000)</li>
  <li><strong>Hardware:</strong> NVIDIA L40S 46GB</li>
</ul>

<h3>Search space</h3>
<pre>learning_rate : loguniform [1e-4, 1e-1]
batch_size    : {{64, 128}}
optimizer     : {{sgd, adam, rmsprop}}
dropout       : uniform [0.1, 0.5]
base_filters  : {{16, 32, 64}}</pre>

<h2>3. Hasil Komparasi 5 Metode</h2>
{comp_table}

{figure_block("Best val acc per metode", "best_val_acc_per_method.png",
              "Akurasi terbaik tiap metode HPO. Hyperband/ASHA tertinggi.")}
{figure_block("Total waktu per metode", "total_time_per_method.png",
              "Total wall-clock time tiap metode. ASHA paling efisien karena pruning early-stop.")}
{figure_block("Konvergensi best-so-far", "convergence_best_so_far.png",
              "Trajectory akurasi terbaik sejauh trial ke-i. ASHA & TPE menanjak lebih cepat.")}
{figure_block("Learning curve best trial", "learning_curve_best_trials.png",
              "Validation accuracy per epoch dari best trial tiap metode.")}
{figure_block("Scatter LR vs Acc", "scatter_lr_vs_acc.png",
              "Hubungan learning rate vs val_acc — sweet spot di sekitar 1e-3.")}

<h2>4. Detail Trial per Metode</h2>
<p>Klik tiap metode untuk melihat semua hyperparameter yang dicoba:</p>
{methods_html}

<h2>5. Final Training (Best Config)</h2>
<div class="key">
  <strong>Best config (dari Hyperband/ASHA):</strong>
  <pre>{json.dumps(params, indent=2)}</pre>
</div>

<h3>Learning curve final (25 epoch)</h3>
{figure_block("Final learning curves", "final_learning_curves.png",
              "Train vs Val accuracy/loss. Tidak overfit signifikan, val masih ikut naik.")}

<details>
  <summary>📋 Tabel learning curve lengkap</summary>
  <table class="data-table">
    <thead><tr><th>Epoch</th><th>Train Loss</th><th>Train Acc</th><th>Val Loss</th><th>Val Acc</th></tr></thead>
    <tbody>{lc_rows}</tbody>
  </table>
</details>

<h3>Confusion matrix di test set</h3>
{figure_block("Confusion matrix", "final_confusion_matrix.png",
              "Confusion matrix 10 kelas. Cat & Dog paling sering tertukar (typical CIFAR-10).")}

<h3>Per-class metrics</h3>
<table class="data-table">
  <thead><tr><th>Kelas</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
  <tbody>{cls_rows}</tbody>
</table>

<div class="note">
  <strong>📝 Catatan untuk diri sendiri:</strong>
  <ul>
    <li>Cat (F1=0.80) adalah kelas terlemah → coba data augmentation lebih agresif (cutmix, mixup).</li>
    <li>Belum pakai LR scheduler — bisa <em>cosine annealing</em> untuk dorong val_acc &gt; 92%.</li>
    <li>Coba juga ResNet-18 sebagai backbone vs custom CNN.</li>
    <li>Untuk laporan dosen: cukup pakai versi <code>laporan_dosen.html</code>.</li>
  </ul>
</div>

<h2>6. Cara Reproduce</h2>
<pre>pip install -r requirements.txt
python scripts/run_all_hpo.py        # 5 metode HPO
python scripts/analyze_results.py    # grafik + tabel
python scripts/final_train.py        # retrain best config
python scripts/build_report.py       # laporan .docx
python scripts/build_html_reports.py # laporan HTML (file ini)</pre>

<hr>
<p style="text-align:center;color:#9ca3af;font-size:.85rem;">
  Generated dari hasil eksperimen di <code>./results/</code> · MLflow UI: <code>mlflow ui --backend-store-uri ./mlruns</code>
</p>

</body>
</html>"""


# ============================================================
# Versi 2: DOSEN (formal, ringkas, akademik)
# ============================================================
def build_dosen(best, final, all_hpo, df) -> str:
    winner = final["winner_method"]
    params = final["params"]
    test_acc = final["test_acc"]

    comp_table = table_from_df(df)

    cr = final["classification_report"]
    classes = [k for k in cr if k not in ("accuracy", "macro avg", "weighted avg")]
    cls_rows = "".join(
        f"<tr><td>{c}</td><td>{cr[c]['precision']:.3f}</td>"
        f"<td>{cr[c]['recall']:.3f}</td><td>{cr[c]['f1-score']:.3f}</td>"
        f"<td>{int(cr[c]['support'])}</td></tr>"
        for c in classes
    )

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>Laporan HPO CIFAR-10</title>
<style>
  @page {{ margin: 2cm; }}
  body {{ font-family: "Times New Roman", Georgia, serif;
          max-width: 900px; margin: 2.5rem auto; padding: 0 2rem;
          color: #111827; line-height: 1.7; background: white; }}
  h1 {{ text-align: center; font-size: 1.6rem; margin-bottom: .3rem; }}
  .subtitle {{ text-align: center; color: #4b5563; margin-bottom: 2rem; font-style: italic; }}
  h2 {{ font-size: 1.25rem; margin-top: 2rem; border-bottom: 1px solid #d1d5db; padding-bottom: .3rem; }}
  h3 {{ font-size: 1.05rem; margin-top: 1.5rem; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: .92rem; }}
  table.data-table th, table.data-table td {{ border: 1px solid #9ca3af; padding: .45rem .6rem; }}
  table.data-table th {{ background: #f3f4f6; }}
  figure {{ margin: 1.2rem 0; text-align: center; page-break-inside: avoid; }}
  figure img {{ max-width: 95%; border: 1px solid #d1d5db; }}
  figcaption {{ color: #4b5563; font-size: .9rem; margin-top: .4rem; font-style: italic; }}
  pre {{ background: #f9fafb; border: 1px solid #e5e7eb; padding: .8rem;
         border-radius: 4px; font-size: .85rem; overflow-x: auto; }}
  .abstrak {{ background: #f9fafb; border-left: 4px solid #1f2937; padding: 1rem 1.2rem;
              margin: 1.5rem 0; font-size: .95rem; }}
  .meta {{ text-align: center; color: #6b7280; font-size: .9rem; }}
  ol li, ul li {{ margin: .3rem 0; }}
</style>
</head>
<body>

<h1>Hyperparameter Optimization untuk Image Classification<br>
Menggunakan Deep Learning Experiment Manager</h1>
<p class="subtitle">Studi Komparasi 5 Metode HPO pada CNN untuk Dataset CIFAR-10</p>

<div class="meta">
  Framework: PyTorch 2.4 · Experiment Manager: MLflow 2.9 · Dataset: CIFAR-10 · Hardware: NVIDIA L40S
</div>

<div class="abstrak">
  <strong>Abstrak.</strong> Penelitian ini membandingkan lima metode <em>Hyperparameter
  Optimization</em> (HPO) — Grid Search, Random Search, Bayesian Optimization (TPE),
  Hyperband/ASHA, dan Genetic Algorithm — pada permasalahan klasifikasi citra CIFAR-10
  menggunakan arsitektur Convolutional Neural Network (CNN) yang diimplementasikan dengan
  PyTorch. MLflow digunakan sebagai <em>experiment manager</em> untuk pelacakan trial,
  parameter, dan metrik. Hasil menunjukkan bahwa metode <strong>{winner}</strong> memberikan
  performa validasi terbaik sebesar <strong>{fmt_pct(best[winner]['val_acc'])}</strong> dalam
  waktu komputasi yang lebih singkat dibandingkan metode lain. Setelah dilakukan pelatihan
  ulang menggunakan konfigurasi terbaik selama 25 epoch, model mencapai akurasi
  <strong>{fmt_pct(test_acc)}</strong> pada test set.
</div>

<h2>1. Pendahuluan</h2>
<p>Pemilihan hyperparameter pada deep learning sangat memengaruhi akurasi dan waktu komputasi
model. Penelitian ini bertujuan membandingkan lima metode HPO dari segi
(i) akurasi validasi terbaik, (ii) total waktu komputasi, dan (iii) jumlah trial.
Dataset yang digunakan adalah CIFAR-10 yang terdiri dari 60.000 citra berukuran 32×32 piksel
yang terbagi atas 10 kelas.</p>

<h2>2. Metodologi</h2>

<h3>2.1 Arsitektur Model</h3>
<p>Custom CNN dengan tiga blok konvolusi (Conv–BN–ReLU–MaxPool) yang diakhiri dengan dua
layer <em>fully connected</em>. Jumlah filter dasar (<code>base_filters</code>) merupakan
hyperparameter yang dioptimalkan.</p>

<h3>2.2 Search Space</h3>
<pre>learning_rate : loguniform [1e-4, 1e-1]
batch_size    : {{64, 128}}
optimizer     : {{SGD, Adam, RMSProp}}
dropout       : uniform [0.1, 0.5]
base_filters  : {{16, 32, 64}}</pre>

<h3>2.3 Setup HPO</h3>
<ul>
  <li>Setiap trial dilatih selama 8 epoch dengan validation set 5.000 sampel.</li>
  <li>MLflow digunakan untuk mencatat parameter dan metrik setiap run.</li>
  <li>Setelah HPO selesai, konfigurasi terbaik dilatih ulang selama 25 epoch pada full train set (45.000).</li>
</ul>

<h2>3. Hasil dan Pembahasan</h2>

<h3>3.1 Komparasi Lima Metode HPO</h3>
{comp_table}

{figure_block("Best val acc per metode", "best_val_acc_per_method.png",
              "Gambar 1. Akurasi validasi terbaik untuk setiap metode HPO.")}

{figure_block("Total waktu per metode", "total_time_per_method.png",
              "Gambar 2. Total waktu komputasi (detik) untuk setiap metode HPO.")}

{figure_block("Konvergensi best-so-far", "convergence_best_so_far.png",
              "Gambar 3. Konvergensi akurasi terbaik sejauh trial ke-i.")}

{figure_block("Learning curve trial terbaik", "learning_curve_best_trials.png",
              "Gambar 4. Learning curve trial terbaik dari masing-masing metode.")}

<h3>3.2 Pembahasan</h3>
<p>Berdasarkan Tabel 1, metode <strong>{winner}</strong> mencapai akurasi validasi
tertinggi sebesar <strong>{fmt_pct(best[winner]['val_acc'])}</strong> dengan total waktu
{best[winner]['total_time_method']:.1f} detik. Metode ini efektif karena mekanisme
<em>early-stopping</em> bertingkat (successive halving) memungkinkan eksplorasi lebih banyak
konfigurasi dengan biaya komputasi yang sama. Bayesian Optimization (TPE) berada pada
peringkat kedua karena memanfaatkan informasi posterior dari trial sebelumnya. Random Search
menunjukkan performa lebih rendah karena keterbatasan budget trial, sementara Genetic
Algorithm membutuhkan waktu paling lama akibat jumlah evaluasi populasi yang besar.</p>

<h3>3.3 Pelatihan Final dengan Konfigurasi Terbaik</h3>
<p>Konfigurasi terbaik yang diperoleh:</p>
<pre>{json.dumps(params, indent=2)}</pre>

<p>Model dilatih ulang selama 25 epoch pada full train set, kemudian dievaluasi pada test set
yang berisi 10.000 sampel.</p>

{figure_block("Final learning curves", "final_learning_curves.png",
              "Gambar 5. Learning curve pelatihan final 25 epoch.")}

{figure_block("Confusion matrix", "final_confusion_matrix.png",
              "Gambar 6. Confusion matrix pada test set CIFAR-10.")}

<h3>3.4 Metrik per Kelas pada Test Set</h3>
<table class="data-table">
  <thead><tr><th>Kelas</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
  <tbody>{cls_rows}</tbody>
</table>

<p>Akurasi total pada test set sebesar <strong>{fmt_pct(test_acc)}</strong> dengan
loss {final['test_loss']:.4f}. Kelas <em>cat</em> memiliki F1-score terendah, yang
sejalan dengan karakteristik CIFAR-10 di mana kelas hewan kecil (cat, dog, bird) sering
saling tertukar akibat kemiripan visual.</p>

<h2>4. Kesimpulan</h2>
<ol>
  <li>Lima metode HPO berhasil dibandingkan secara terkendali pada permasalahan klasifikasi CIFAR-10.</li>
  <li>Metode <strong>{winner}</strong> memberikan trade-off terbaik antara akurasi
      ({fmt_pct(best[winner]['val_acc'])}) dan efisiensi waktu
      ({best[winner]['total_time_method']:.1f} detik).</li>
  <li>Pelatihan ulang dengan konfigurasi terbaik selama 25 epoch menghasilkan akurasi test
      <strong>{fmt_pct(test_acc)}</strong>, menunjukkan bahwa hyperparameter hasil HPO dapat
      digeneralisasi dengan baik pada data uji.</li>
  <li>MLflow terbukti efektif sebagai <em>experiment manager</em> untuk pelacakan,
      reproduksi, dan komparasi banyak trial secara terstruktur.</li>
</ol>

<h2>Daftar Pustaka</h2>
<ol>
  <li>Bergstra, J., & Bengio, Y. (2012). <em>Random search for hyper-parameter optimization</em>. JMLR, 13, 281–305.</li>
  <li>Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). <em>Algorithms for hyper-parameter optimization</em>. NIPS.</li>
  <li>Li, L., et al. (2018). <em>Hyperband: A novel bandit-based approach to hyperparameter optimization</em>. JMLR, 18, 1–52.</li>
  <li>Akiba, T., et al. (2019). <em>Optuna: A next-generation hyperparameter optimization framework</em>. KDD.</li>
  <li>Krizhevsky, A. (2009). <em>Learning multiple layers of features from tiny images</em> (CIFAR-10 Technical Report).</li>
  <li>Zaharia, M., et al. (2018). <em>Accelerating the Machine Learning Lifecycle with MLflow</em>. IEEE Data Engineering Bulletin.</li>
</ol>

</body>
</html>"""


def main():
    best, final, all_hpo, df = load_data()

    belajar_path = OUT_DIR / "laporan_belajar.html"
    dosen_path = OUT_DIR / "laporan_dosen.html"

    belajar_path.write_text(build_belajar(best, final, all_hpo, df), encoding="utf-8")
    dosen_path.write_text(build_dosen(best, final, all_hpo, df), encoding="utf-8")

    print(f"✅ Laporan belajar : {belajar_path}  ({belajar_path.stat().st_size//1024} KB)")
    print(f"✅ Laporan dosen   : {dosen_path}    ({dosen_path.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()

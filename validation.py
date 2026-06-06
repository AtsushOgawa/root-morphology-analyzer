"""
validation.py  v2.0
====================
root_analysis.py (result.csv) と WinRhizo の出力 CSV を統計的に比較・検証するスクリプト

【出力ファイル】
  validation_stats.csv   : 各指標の統計サマリー（相関・回帰・Bland-Altman など）
  validation_data.csv    : サンプルごとの対照表（スクリプト値・WinRhizo値・差・相対誤差）
  validation_plots.pdf   : 散布図 + Bland-Altman プロット（5指標 × 2グラフ）

【統計手法】
  - Pearson 相関係数 (r) と有意確率
  - 線形回帰（傾き・切片・R²）
  - 対応のある t 検定（平均値の有意差）
  - RMSE・MAE・平均相対誤差(%)
  - Bland-Altman 解析（バイアス・一致限界 ±1.96SD）

【動作要件】
  Python 3.8 以上
  pip install numpy pandas scipy matplotlib
"""

import csv
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

warnings.filterwarnings("ignore")


# ================================================================
# ★ パス設定（必要に応じて変更してください）
# ================================================================

# スクリプト結果 CSV（root_analysis.py が出力する result.csv）
SCRIPT_CSV   = Path.home() / "Desktop" / "根画像" / "result.csv"

# WinRhizo 出力 CSV
WINRHIZO_CSV = Path.home() / "Desktop" / "winrhizo_result.csv"

# 出力先フォルダ
OUTPUT_DIR   = Path.home() / "Desktop" / "根画像"


# ================================================================
# ★ WinRhizo CSV の列名設定
#    WinRhizo のエクスポート列名に合わせて変更してください。
#    設定しなくても起動時に自動検出を試みます。
#    自動検出が失敗した場合は、起動時に表示される「列名一覧」から
#    該当する列名を以下に貼り付けてください。
# ================================================================

# WinRhizo のファイル名列（画像ファイル名が入っている列）
WINRHIZO_FILENAME_COL = "Image"

# WinRhizo 各指標の列名
WINRHIZO_COLUMNS = {
    "total_length_mm" : "Length",    # 総根長
    "tip_count"       : "Tips",      # 根端数
    "surface_cm2"     : "SurfArea",  # 根表面積
    "mean_diam_mm"    : "AvgDiam",   # 根平均直径
    "volume_cm3"      : "Volume",    # 根の体積
}

# ★ WinRhizo の根長単位（"mm" または "cm"）
#    WinRhizo はデフォルトで cm 出力の場合があります。
#    cm で出力されている場合は "cm" に変更してください。
WINRHIZO_LENGTH_UNIT = "cm"   # "mm" or "cm"


# ================================================================
# スクリプト結果 CSV の列名（通常変更不要）
# ================================================================

SCRIPT_FILENAME_COL = "ファイル名"

SCRIPT_COLUMNS = {
    "total_length_mm" : "総根長(mm)",
    "tip_count"       : "根端数",
    "surface_cm2"     : "根表面積(cm2)",
    "mean_diam_mm"    : "根平均直径(mm)",
    "volume_cm3"      : "根の体積(cm3)",
}

# 表示用ラベル
LABELS = {
    "total_length_mm" : "総根長 (mm)",
    "tip_count"       : "根端数",
    "surface_cm2"     : "根表面積 (cm²)",
    "mean_diam_mm"    : "根平均直径 (mm)",
    "volume_cm3"      : "根の体積 (cm³)",
}


# ================================================================
# 列名自動検出（大文字小文字・スペース・記号を無視した照合）
# ================================================================

# WinRhizo のバージョンや言語設定によって変わりうる列名の候補一覧
_FILENAME_CANDIDATES = [
    "Image", "image", "Filename", "filename", "FileName",
    "File", "file", "file name", "File name", "ImageFile",
    "Name", "name", "Sample", "SampleID", "sample",
]

_METRIC_CANDIDATES = {
    "total_length_mm": [
        "Length", "length", "Total Length", "TotalLength",
        "Root Length", "RootLength", "Root length",
        "Length cm", "Length(cm)", "Length (cm)",
        "Length mm", "Length(mm)", "Length (mm)",
        "根長", "総根長",
    ],
    "tip_count": [
        "Tips", "tips", "Root Tips", "RootTips",
        "Tip Count", "TipCount", "Nb tips", "NbTips",
        "Number of tips", "根端数", "先端数",
    ],
    "surface_cm2": [
        "SurfArea", "Surf.area", "Surface area", "SurfaceArea",
        "Surface Area", "Surface", "surface area",
        "Surface area cm2", "Surf Area", "根表面積", "表面積",
    ],
    "mean_diam_mm": [
        "AvgDiam", "Avg Diam", "Avg.Diam", "Average Diameter",
        "AvgDiameter", "Mean Diam", "MeanDiam", "Mean diam",
        "Avg diam", "Diameter", "diameter", "根平均直径", "平均直径",
    ],
    "volume_cm3": [
        "Volume", "volume", "Root Volume", "RootVolume",
        "Root volume", "Vol", "体積", "根の体積",
    ],
}


def _normalize(s: str) -> str:
    """大文字小文字・スペース・ピリオドを除去した比較用文字列"""
    return s.lower().replace(" ", "").replace(".", "").replace("_", "")


def _find_col(candidates: list, available: list) -> str | None:
    """候補リストと実際の列名リストを照合して最初にヒットした列名を返す"""
    avail_norm = {_normalize(c): c for c in available}
    # 完全一致を優先
    for cand in candidates:
        if cand in available:
            return cand
    # 正規化一致
    for cand in candidates:
        hit = avail_norm.get(_normalize(cand))
        if hit:
            return hit
    # 部分一致（候補が列名に含まれる）
    for cand in candidates:
        nc = _normalize(cand)
        for norm_col, orig_col in avail_norm.items():
            if nc in norm_col or norm_col in nc:
                return orig_col
    return None


# ================================================================
# 統計計算
# ================================================================

def compute_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """
    x : WinRhizo の測定値
    y : スクリプトの測定値
    """
    n = len(x)
    r, p_r = stats.pearsonr(x, y)
    slope, intercept, r_value, p_reg, std_err = stats.linregress(x, y)
    t_stat, p_t = stats.ttest_rel(x, y)

    diff    = y - x
    rmse    = math.sqrt(float(np.mean(diff ** 2)))
    mae     = float(np.mean(np.abs(diff)))
    rel_err = float(np.nanmean(diff / np.where(x != 0, x, np.nan))) * 100

    ba_mean = (x + y) / 2.0
    ba_diff = y - x
    ba_bias = float(np.mean(ba_diff))
    ba_sd   = float(np.std(ba_diff, ddof=1))
    ba_lo   = ba_bias - 1.96 * ba_sd
    ba_hi   = ba_bias + 1.96 * ba_sd

    return dict(
        n=n, r=float(r), p_r=float(p_r),
        slope=float(slope), intercept=float(intercept),
        R2=float(r_value**2), p_reg=float(p_reg),
        t_stat=float(t_stat), p_t=float(p_t),
        RMSE=rmse, MAE=mae, mean_rel_err=rel_err,
        ba_bias=ba_bias, ba_sd=ba_sd, ba_lo=ba_lo, ba_hi=ba_hi,
        ba_x=ba_mean, ba_y=ba_diff,
    )


# ================================================================
# プロット
# ================================================================

def _sig(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def plot_scatter(ax, x, y, sd, label):
    ax.scatter(x, y, s=38, alpha=0.75, color="#2171b5",
               edgecolors="white", linewidths=0.5, zorder=3)

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    mg = (hi - lo) * 0.06
    lim = (lo - mg, hi + mg)

    ax.plot(lim, lim, "k--", lw=1.2, alpha=0.45, label="y = x (一致線)")
    xfit = np.linspace(lim[0], lim[1], 300)
    ax.plot(xfit, sd["slope"]*xfit + sd["intercept"], "-",
            color="#d62728", lw=1.8,
            label=f"回帰: y={sd['slope']:.3f}x{sd['intercept']:+.3f}")

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"WinRhizo  {label}", fontsize=9)
    ax.set_ylabel(f"スクリプト  {label}", fontsize=9)
    ax.set_title(
        f"{label}\nr={sd['r']:.4f}{_sig(sd['p_r'])},  R²={sd['R2']:.4f},  n={sd['n']}",
        fontsize=9, fontweight="bold"
    )
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.22, ls=":")
    ax.text(0.97, 0.04,
            f"平均相対誤差: {sd['mean_rel_err']:+.2f}%\n"
            f"RMSE: {sd['RMSE']:.4g}\n"
            f"対応t検定 p={sd['p_t']:.4f}{_sig(sd['p_t'])}",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.85))


def plot_bland_altman(ax, sd, label):
    ax.scatter(sd["ba_x"], sd["ba_y"], s=38, alpha=0.75, color="#fd8d3c",
               edgecolors="white", linewidths=0.5, zorder=3)
    ax.axhline(sd["ba_bias"], color="#1a1a1a", lw=1.8,
               label=f"バイアス: {sd['ba_bias']:.4g}")
    ax.axhline(sd["ba_hi"],   color="#1a1a1a", lw=1.2, ls="--",
               label=f"+1.96SD: {sd['ba_hi']:.4g}")
    ax.axhline(sd["ba_lo"],   color="#1a1a1a", lw=1.2, ls="--",
               label=f"−1.96SD: {sd['ba_lo']:.4g}")
    ax.axhline(0, color="steelblue", lw=0.9, ls=":", alpha=0.6)
    ax.set_xlabel("平均値  (WinRhizo + スクリプト) / 2", fontsize=9)
    ax.set_ylabel("差  (スクリプト − WinRhizo)", fontsize=9)
    ax.set_title(
        f"Bland-Altman  {label}\n"
        f"LoA = [{sd['ba_lo']:.4g},  {sd['ba_hi']:.4g}]",
        fontsize=9, fontweight="bold"
    )
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True, alpha=0.22, ls=":")


def make_summary_page(pdf, all_stats):
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle("検証サマリー：スクリプト vs WinRhizo  全指標",
                 fontsize=13, fontweight="bold", y=0.99)
    ax = fig.add_subplot(111)
    ax.axis("off")

    cols = ["指標", "n", "r", "R²", "傾き", "切片",
            "平均相対誤差(%)", "RMSE", "MAE",
            "t検定 p値", "BA バイアス", "LoA 下限", "LoA 上限"]
    rows = []
    for metric, sd in all_stats.items():
        rows.append([
            LABELS[metric], sd["n"],
            f"{sd['r']:.4f}{_sig(sd['p_r'])}",
            f"{sd['R2']:.4f}",
            f"{sd['slope']:.4f}", f"{sd['intercept']:.4f}",
            f"{sd['mean_rel_err']:+.2f}",
            f"{sd['RMSE']:.4g}", f"{sd['MAE']:.4g}",
            f"{sd['p_t']:.4f}{_sig(sd['p_t'])}",
            f"{sd['ba_bias']:.4g}",
            f"{sd['ba_lo']:.4g}", f"{sd['ba_hi']:.4g}",
        ])

    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.1)

    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2c7bb6")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)+1):
        bg = "#f0f8ff" if i % 2 == 0 else "white"
        _, sd = list(all_stats.items())[i-1]
        if sd["p_t"] < 0.05:
            bg = "#fff3cd" if i % 2 == 0 else "#fffde7"
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(bg)

    fig.text(0.01, 0.01,
             "*** p<0.001  ** p<0.01  * p<0.05  n.s. 有意差なし　"
             "｜　黄色行：スクリプトとWinRhizoの間に有意差あり（t検定 p<0.05）",
             fontsize=7, color="#555")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# メイン処理
# ================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sep = "=" * 70

    print(sep)
    print("  根形態パラメータ検証スクリプト  v2.0")
    print(sep)

    # ── スクリプト結果 CSV 読み込み ───────────────────────────
    try:
        df_scr = pd.read_csv(SCRIPT_CSV, encoding="utf-8-sig")
        print(f"スクリプト結果: {len(df_scr)} 行")
    except Exception as e:
        print(f"[エラー] スクリプト結果CSV 読み込み失敗: {e}")
        return

    # ── WinRhizo CSV 読み込み（複数エンコーディングを試みる）─
    df_wz = None
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis", "latin-1"):
        try:
            df_wz = pd.read_csv(WINRHIZO_CSV, encoding=enc)
            print(f"WinRhizo 結果 : {len(df_wz)} 行  (encoding={enc})")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"[エラー] WinRhizo CSV 読み込み失敗: {e}")
            return
    if df_wz is None:
        print("[エラー] WinRhizo CSV をどのエンコーディングでも読み込めませんでした。")
        return

    print(f"\nWinRhizo 列名一覧: {list(df_wz.columns)}\n")

    # ── WinRhizo ファイル名列の自動検出 ──────────────────────
    fn_col = _find_col([WINRHIZO_FILENAME_COL] + _FILENAME_CANDIDATES,
                       list(df_wz.columns))
    if fn_col is None:
        print("[エラー] WinRhizo CSV にファイル名列が見つかりません。")
        print("  → WINRHIZO_FILENAME_COL を上記の列名一覧から選んで設定してください。")
        return
    if fn_col != WINRHIZO_FILENAME_COL:
        print(f"[自動検出] ファイル名列: '{WINRHIZO_FILENAME_COL}' → '{fn_col}'")

    # ── WinRhizo 各指標列の自動検出 ──────────────────────────
    # 検出した列名を固定内部名（_wz_XXX）に変換してからマージするため
    # マージ後の列名衝突・サフィックス問題を完全に回避する
    col_map = {}   # metric -> WinRhizo実際の列名
    for metric in LABELS:
        # 設定値を最優先候補の先頭に追加して検索
        candidates = [WINRHIZO_COLUMNS.get(metric, "")] + _METRIC_CANDIDATES.get(metric, [])
        found = _find_col(candidates, list(df_wz.columns))
        if found:
            col_map[metric] = found
            status = "（設定値）" if found == WINRHIZO_COLUMNS.get(metric) else f"（自動検出: '{WINRHIZO_COLUMNS.get(metric)}' → '{found}'）"
            print(f"  {LABELS[metric]:20s}: {found}  {status}")
        else:
            print(f"  {LABELS[metric]:20s}: [未検出] ← WINRHIZO_COLUMNS['{metric}'] を設定してください")

    found_count = len(col_map)
    if found_count == 0:
        print("\n[エラー] WinRhizo の指標列が1つも見つかりません。スクリプト冒頭の WINRHIZO_COLUMNS を設定してください。")
        return
    print(f"\n指標列検出: {found_count}/5 件\n")

    # ── WinRhizo 根長の単位変換（cm → mm）──────────────────
    if WINRHIZO_LENGTH_UNIT == "cm" and "total_length_mm" in col_map:
        df_wz[col_map["total_length_mm"]] = (
            pd.to_numeric(df_wz[col_map["total_length_mm"]], errors="coerce") * 10.0
        )
        print(f"[変換] WinRhizo 根長: cm → mm (×10)")

    # ── WinRhizo 側の列を固定内部名にリネームしてから stem を付与 ──
    # 内部名: _wz_total_length_mm, _wz_tip_count, ...
    wz_rename = {fn_col: "_stem_wz"}
    for metric, orig_col in col_map.items():
        wz_rename[orig_col] = f"_wz_{metric}"
    df_wz2 = df_wz.rename(columns=wz_rename)[list(wz_rename.values())].copy()
    df_wz2["_stem"] = df_wz2["_stem_wz"].apply(lambda s: Path(str(s)).stem.strip())

    # ── スクリプト側も stem を付与 ───────────────────────────
    df_scr["_stem"] = df_scr[SCRIPT_FILENAME_COL].apply(
        lambda s: Path(str(s)).stem.strip()
    )

    # ── マージ（_stem キーで内部結合）────────────────────────
    df = pd.merge(df_scr, df_wz2, on="_stem")
    print(f"マッチしたサンプル数: {len(df)}  "
          f"（スクリプト {len(df_scr)}, WinRhizo {len(df_wz)} 行）")

    if len(df) == 0:
        print("\n[エラー] マッチするサンプルがありません。")
        print(f"  スクリプト stem 例: {list(df_scr['_stem'].head(5))}")
        print(f"  WinRhizo  stem 例: {list(df_wz2['_stem'].head(5))}")
        return

    # ── エラー行を除外 ────────────────────────────────────────
    scr_len_col = SCRIPT_COLUMNS["total_length_mm"]
    if scr_len_col in df.columns:
        err_mask = df[scr_len_col].astype(str).str.upper() == "ERROR"
        if err_mask.sum() > 0:
            print(f"[除外] スクリプトエラー行 {err_mask.sum()} 件")
        df = df[~err_mask].copy()
    print(f"有効サンプル数: {len(df)}\n")

    # ── matplotlib 日本語フォント ─────────────────────────────
    plt.rcParams["font.family"] = [
        "IPAexGothic", "IPAGothic", "Noto Sans CJK JP",
        "Hiragino Sans", "Yu Gothic", "Meiryo", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # ── 各指標を順にプロット・統計計算 ───────────────────────
    all_stats = {}
    pdf_path  = OUTPUT_DIR / "validation_plots.pdf"

    with mpdf.PdfPages(pdf_path) as pdf:
        for metric in LABELS:
            if metric not in col_map:
                print(f"[スキップ] {LABELS[metric]}: WinRhizo 列が未検出")
                continue

            scr_col = SCRIPT_COLUMNS[metric]
            wz_col  = f"_wz_{metric}"          # 固定内部名

            if scr_col not in df.columns:
                print(f"[スキップ] {LABELS[metric]}: スクリプト列 '{scr_col}' が見つかりません")
                continue

            x = pd.to_numeric(df[wz_col],  errors="coerce").values.astype(float)
            y = pd.to_numeric(df[scr_col], errors="coerce").values.astype(float)
            valid = ~(np.isnan(x) | np.isnan(y))
            x, y  = x[valid], y[valid]

            if len(x) < 3:
                print(f"[スキップ] {LABELS[metric]}: 有効データ {len(x)} 件（3件以上必要）")
                continue

            sd = compute_stats(x, y)
            all_stats[metric] = sd

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
            fig.suptitle(f"検証: {LABELS[metric]}  (n={sd['n']})",
                         fontsize=12, fontweight="bold", y=1.01)
            plot_scatter(ax1, x, y, sd, LABELS[metric])
            plot_bland_altman(ax2, sd, LABELS[metric])
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            print(f"── {LABELS[metric]} {'─'*40}")
            print(f"   n={sd['n']}")
            print(f"   Pearson r={sd['r']:.4f}  p={sd['p_r']:.4f} {_sig(sd['p_r'])}")
            print(f"   R²={sd['R2']:.4f}")
            print(f"   回帰: 傾き={sd['slope']:.4f}  切片={sd['intercept']:.4f}")
            print(f"   平均相対誤差: {sd['mean_rel_err']:+.2f}%")
            print(f"   RMSE={sd['RMSE']:.4g}  MAE={sd['MAE']:.4g}")
            print(f"   対応t検定: t={sd['t_stat']:.4f}  p={sd['p_t']:.4f} {_sig(sd['p_t'])}")
            print(f"   Bland-Altman: バイアス={sd['ba_bias']:.4g}  "
                  f"LoA=[{sd['ba_lo']:.4g}, {sd['ba_hi']:.4g}]")
            print()

        if all_stats:
            make_summary_page(pdf, all_stats)

    # ── 統計サマリー CSV ──────────────────────────────────────
    if all_stats:
        stat_rows = []
        for metric, sd in all_stats.items():
            stat_rows.append({
                "指標"            : LABELS[metric],
                "n"               : sd["n"],
                "Pearson r"       : round(sd["r"], 4),
                "p(r)"            : round(sd["p_r"], 4),
                "有意性(r)"       : _sig(sd["p_r"]),
                "R²"              : round(sd["R2"], 4),
                "回帰_傾き"       : round(sd["slope"], 4),
                "回帰_切片"       : round(sd["intercept"], 4),
                "平均相対誤差(%)" : round(sd["mean_rel_err"], 3),
                "RMSE"            : round(sd["RMSE"], 4),
                "MAE"             : round(sd["MAE"], 4),
                "t統計量"         : round(sd["t_stat"], 4),
                "p(t検定)"        : round(sd["p_t"], 4),
                "有意性(t)"       : _sig(sd["p_t"]),
                "BA_バイアス"     : round(sd["ba_bias"], 4),
                "BA_SD"           : round(sd["ba_sd"], 4),
                "BA_LoA下限"      : round(sd["ba_lo"], 4),
                "BA_LoA上限"      : round(sd["ba_hi"], 4),
            })
        stats_csv = OUTPUT_DIR / "validation_stats.csv"
        with open(stats_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
            w.writeheader(); w.writerows(stat_rows)
        print(f"統計サマリー保存: {stats_csv}")

    # ── サンプル別対照表 CSV ──────────────────────────────────
    data_rows = []
    for _, row in df.iterrows():
        entry = {"ファイル名": row["_stem"]}
        for metric in LABELS:
            if metric not in col_map:
                continue
            scr_col = SCRIPT_COLUMNS[metric]
            wz_col  = f"_wz_{metric}"
            sv = pd.to_numeric(row.get(scr_col), errors="coerce")
            wv = pd.to_numeric(row.get(wz_col),  errors="coerce")
            entry[f"{LABELS[metric]}_スクリプト"] = round(float(sv), 4) if pd.notna(sv) else ""
            entry[f"{LABELS[metric]}_WinRhizo"]   = round(float(wv), 4) if pd.notna(wv) else ""
            if pd.notna(sv) and pd.notna(wv) and wv != 0:
                entry[f"{LABELS[metric]}_差"]        = round(float(sv - wv), 4)
                entry[f"{LABELS[metric]}_相対誤差%"] = round(float((sv - wv) / wv * 100), 2)
            else:
                entry[f"{LABELS[metric]}_差"]        = ""
                entry[f"{LABELS[metric]}_相対誤差%"] = ""
        data_rows.append(entry)

    if data_rows:
        data_csv = OUTPUT_DIR / "validation_data.csv"
        with open(data_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data_rows[0].keys()))
            w.writeheader(); w.writerows(data_rows)
        print(f"サンプル別対照表保存: {data_csv}")

    print(f"プロット保存        : {pdf_path}")
    print(f"\n{sep}")
    print(f"  検証完了  {len(all_stats)}/5 指標")
    print(sep)


if __name__ == "__main__":
    main()

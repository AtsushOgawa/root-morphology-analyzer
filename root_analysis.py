"""
root_analysis.py  v4.2
======================
複数の根スキャン画像から根形態パラメータを一括測定するスクリプト

【測定項目】
  1. 総根長 (mm)
  2. 根端数
  3. 根表面積 (cm²)
  4. 根平均直径 (mm)
  5. 根の体積 (cm³)

【自動条件設定の仕組み】
  画像の平均輝度から「染色の有無」を自動判定し、対応するパラメータを自動設定します。

  ┌──────────────────────────────────────────────────────────────────┐
  │  平均輝度 < 128 → 根が明色・背景が暗色                             │
  │            ＝ 染色なし（太めの根：レタスなど）                       │
  │            → BRIGHT 系パラメータを使用                              │
  │                                                                    │
  │  平均輝度 ≥ 128 → 根が暗色・背景が明色                              │
  │            ＝ 染色あり（細めの根：イネなど）                          │
  │            → DARK 系パラメータを使用                                │
  └──────────────────────────────────────────────────────────────────┘

【各設定の背景と較正データ】

  染色なし（明色根）— BRIGHT 系パラメータ
    ・太めの根（レタス等）のスキャン。根毛もスキャンされる。
    ・根毛と側根先端の直径が近いため、プルーニングで長さにより区別。
    ・較正：レタス8枚（WinRhizo比較）
      根端数 mean=-1.4%±15.9%、総根長 mean=+0.4%±6.0%

  染色あり（暗色根）— DARK 系パラメータ
    ・細めの根（イネ等）のスキャン。染色により根毛は見えにくい。
    ・ガウシアンブラーで染色ムラを補正。
    ・較正：ふさC系・初星C系16枚（WinRhizo比較）
      根端数 mean=+4.4%±5.1%、総根長 mean=0.0%±2.2%

【計算式（WinRhizo 互換）】
  総根長 (mm)     = スケルトン画素数 × (25.4/DPI) × 根長補正係数
  根平均直径 (mm) = スケルトン画素の局所直径平均 ÷ 直径補正係数(1.0817)
  根表面積 (cm²)  = π × 補正後平均直径(mm) × 補正後総根長(mm) / 100
  根の体積 (cm³)  = π × (補正後平均直径(mm)/2)² × 補正後総根長(mm) / 1000
  根端数          = プルーニング後、局所直径が DIAM_LO 以上 DIAM_HI 未満の端点数

【主な設定】
  入力フォルダ : デスクトップの「根画像」フォルダ（サブフォルダも対象）
  解像度       : 画像メタデータから自動取得（取得できない場合は DEFAULT_DPI）
  2値化        : Otsu 法（自動）
  出力         : 「根画像」フォルダ内の result.csv

  ★ 新しい植物種・スキャン条件では、WinRhizo との比較による再較正を推奨
     特に PRUNE_MM と LENGTH_CORR が植物種・スキャン設定に依存します

【動作要件】
  Python 3.8 以上
  pip install numpy pillow scikit-image scipy
"""

import csv
import math
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.signal import convolve2d
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, skeletonize

warnings.filterwarnings("ignore")


# ================================================================
# ★ パラメータ設定（必要に応じて変更してください）
# ================================================================

DEFAULT_DPI         = 1200      # メタデータから取得できない場合のデフォルト解像度 (dpi)
MIN_OBJECT_PX       = 500       # ノイズ除去：この画素数未満の連結成分を除去

# ── 根長補正係数 ──────────────────────────────────────────────────
# スケルトン法の斜め移動補正 + スキャン条件・根の特性による補正値
# 新しいスキャン条件では WinRhizo との比較で再較正を推奨
BRIGHT_LENGTH_CORR  = 1.0775    # 染色なし（太根・根毛あり）: 50枚検証データで再較正
DARK_LENGTH_CORR    = 1.1491    # 染色あり（細根・根毛なし）: 49枚検証データで再較正

# ── 平均直径補正係数（極性別）────────────────────────────────────
# スケルトン法 → WinRhizo ヒストグラム法への換算係数
# 99枚の検証データより極性別に最適値を設定
BRIGHT_DIAM_CORR    = 1.0287    # 染色なし: 50枚検証データで較正
DARK_DIAM_CORR      = 1.1845    # 染色あり: 49枚検証データで較正

# ── 染色なし（明色根）設定 ─────────────────────────────────────────
# 太めの根をスキャン。根毛も撮影される。
# 根毛と側根先端が近い直径のため、プルーニング（枝長）で区別する。
# 較正データ：レタス8枚（WinRhizo比較）
BRIGHT_SIGMA        = 0         # Gaussian blur sigma（0 = なし）
BRIGHT_PRUNE_MM     = 0.127     # プルーニング閾値 (mm)  レタス8枚で再較正（Automatic条件）
BRIGHT_DIAM_LO      = 0.00      # 根端フィルタ下限直径 (mm)  0.00 = 下限なし
BRIGHT_DIAM_HI      = 9.99      # 根端フィルタ上限直径 (mm)  9.99 = 上限なし

# ── 染色あり（暗色根）設定 ─────────────────────────────────────────
# 細めの根を染色してスキャン。根毛は見えにくい。
# 太い根の染色ムラをガウシアンブラーで補正してから2値化する。
# 較正データ：初星C系・ふさC系16枚（WinRhizo比較）
DARK_SIGMA          = 1         # Gaussian blur sigma（染色ムラ補正）
DARK_PRUNE_MM       = 0.05      # プルーニング閾値 (mm)  ふさC系8枚で再較正（Automatic条件）
DARK_DIAM_LO        = 0.00      # 根端フィルタ下限直径 (mm)  0.00 = 下限なし
DARK_DIAM_HI        = 0.30      # 根端フィルタ上限直径 (mm)  太い根の途切れを除外

# ── 対応画像形式 ───────────────────────────────────────────────────
SUPPORTED_EXT       = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# ================================================================
# 内部関数
# ================================================================

_KERNEL = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]], dtype=np.int32)


def _get_dpi(img: Image.Image) -> float:
    """画像メタデータから DPI を取得。取得できない場合は DEFAULT_DPI を返す。"""
    try:
        info = img.info
        for key in ("dpi", "resolution"):
            if key in info:
                v   = info[key]
                dpi = float(v[0] if isinstance(v, tuple) else v)
                if dpi > 0:
                    return dpi
        exif = img._getexif() if hasattr(img, "_getexif") else None
        if exif:
            xres = exif.get(282)
            unit = exif.get(296, 2)
            if xres is not None:
                val = xres[0] / xres[1] if isinstance(xres, tuple) else float(xres)
                if unit == 3:
                    val *= 2.54
                if val > 0:
                    return float(val)
    except Exception:
        pass
    return float(DEFAULT_DPI)


def _detect_polarity(arr: np.ndarray) -> str:
    """平均輝度から根の極性を判定。'bright' または 'dark' を返す。"""
    return "dark" if arr.mean() >= 128 else "bright"


def _prune_skeleton(skel: np.ndarray, min_spur_px: float) -> np.ndarray:
    """
    スケルトンから短いスパー（偽端点）を反復除去する。
    端点（8近傍隣接スケルトン画素が 1 個）を ceil(min_spur_px)+1 回繰り返し消去。
    """
    n_iter = int(np.ceil(min_spur_px)) + 1
    skel   = skel.copy()
    for _ in range(n_iter):
        nc = convolve2d(skel.astype(np.int32), _KERNEL, mode="same")
        ep = skel & (nc == 1)
        if ep.sum() == 0:
            break
        skel = skel & ~ep
    return skel


# ================================================================
# メイン解析関数
# ================================================================

def analyze_image(img_path: Path) -> dict:
    """
    1 枚の根スキャン画像を解析し、5 項目の測定値を返す。

    Returns
    -------
    dict のキー:
        filename, dpi, polarity, otsu_threshold,
        total_length_mm, tip_count,
        surface_cm2, mean_diam_mm, volume_cm3, error
    """
    result = dict(
        filename        = img_path.name,
        dpi             = DEFAULT_DPI,
        polarity        = None,
        otsu_threshold  = None,
        total_length_mm = None,
        tip_count       = None,
        surface_cm2     = None,
        mean_diam_mm    = None,
        volume_cm3      = None,
        error           = None,
    )

    try:
        # ── 画像読み込み ──────────────────────────────────────
        img        = Image.open(img_path)
        dpi        = _get_dpi(img)
        result["dpi"] = dpi
        mm_per_px  = 25.4 / dpi

        # ── 16ビット画像の正規化（I;16 等は convert(L) で値が壊れるため） ──
        if img.mode in ("I;16", "I;16B", "I;16S", "I;16L"):
            arr16 = np.array(img).astype(np.uint16)
            arr   = (arr16 / 256).astype(np.uint8)
        elif img.mode == "I":
            arr_raw = np.array(img)
            mn, mx  = int(arr_raw.min()), int(arr_raw.max())
            arr = ((arr_raw - mn) / max(mx - mn, 1) * 255).astype(np.uint8)
        else:
            arr = np.array(img.convert("L"))

        # ── 極性判定・パラメータ選択 ─────────────────────────
        polarity = _detect_polarity(arr)
        result["polarity"] = polarity

        if polarity == "bright":
            sigma      = BRIGHT_SIGMA
            prune_mm   = BRIGHT_PRUNE_MM
            diam_lo    = BRIGHT_DIAM_LO
            diam_hi    = BRIGHT_DIAM_HI
            len_corr   = BRIGHT_LENGTH_CORR
        else:
            sigma      = DARK_SIGMA
            prune_mm   = DARK_PRUNE_MM
            diam_lo    = DARK_DIAM_LO
            diam_hi    = DARK_DIAM_HI
            len_corr   = DARK_LENGTH_CORR

        # ── 前処理・Otsu 2値化 ───────────────────────────────
        ap = (gaussian_filter(arr.astype(np.float32), sigma=sigma)
              if sigma > 0 else arr.astype(np.float32))
        thr = threshold_otsu(ap)
        result["otsu_threshold"] = round(float(thr), 1)

        binary = (ap < thr) if polarity == "dark" else (ap > thr)
        binary = remove_small_objects(binary.astype(bool), min_size=MIN_OBJECT_PX)

        # ── スケルトン化・プルーニング ────────────────────────
        skel = skeletonize(binary)
        skel = _prune_skeleton(skel, min_spur_px=prune_mm / mm_per_px)

        # ── 距離変換（各画素の局所半径を求める）──────────────
        dist = distance_transform_edt(binary)

        # ── 総根長 (mm)：スケルトン画素数 × mm/px × 根長補正係数
        total_length_mm = float(skel.sum()) * mm_per_px * len_corr

        # ── 根平均直径 (mm)：スケルトン画素の局所直径平均 ÷ 直径補正係数
        sy, sx = np.where(skel)
        if len(sy) == 0:
            raise ValueError("スケルトン画素が 0 です。閾値や極性を確認してください。")
        raw_diam_mm = float(dist[sy, sx].mean()) * 2.0 * mm_per_px
        diam_corr    = BRIGHT_DIAM_CORR if polarity == 'bright' else DARK_DIAM_CORR
        mean_diam_mm = raw_diam_mm / diam_corr

        # ── 根表面積・根の体積（WinRhizo 互換計算式）─────────
        surface_cm2 = math.pi * mean_diam_mm * total_length_mm / 100.0
        volume_cm3  = math.pi * (mean_diam_mm / 2.0) ** 2 * total_length_mm / 1000.0

        # ── 根端数（局所直径フィルタ付き）────────────────────
        nc = convolve2d(skel.astype(np.int32), _KERNEL, mode="same")
        ep_y, ep_x = np.where(skel & (nc == 1))
        if len(ep_y) > 0:
            diams_ep  = dist[ep_y, ep_x] * 2.0 * mm_per_px
            tip_count = int(((diams_ep >= diam_lo) & (diams_ep < diam_hi)).sum())
        else:
            tip_count = 0

        # ── 結果格納 ─────────────────────────────────────────
        result.update(
            total_length_mm = round(total_length_mm, 3),
            tip_count       = tip_count,
            surface_cm2     = round(surface_cm2, 4),
            mean_diam_mm    = round(mean_diam_mm, 4),
            volume_cm3      = round(volume_cm3, 4),
        )

    except Exception as e:
        result["error"] = str(e)

    return result


# ================================================================
# メイン処理
# ================================================================

def main():
    desktop    = Path.home() / "Desktop"
    img_folder = desktop / "根画像"
    output_csv = img_folder / "result.csv"

    if not img_folder.exists():
        print(f"[エラー] フォルダが見つかりません: {img_folder}")
        return

    img_files = sorted(
        p for p in img_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )

    if not img_files:
        print(f"[警告] 対象画像が見つかりません: {img_folder}")
        return

    sep = "=" * 72
    print(sep)
    print("  根形態パラメータ自動測定スクリプト  v4.2")
    print(sep)
    print(f"  対象フォルダ      : {img_folder}")
    print(f"  対象画像数        : {len(img_files)} 枚")
    print(f"  デフォルト DPI    : {DEFAULT_DPI}")
    print(f"  根長補正係数      : 明色根 {BRIGHT_LENGTH_CORR} / 暗色根 {DARK_LENGTH_CORR}")
    print(f"  直径補正係数      : 染色なし÷{BRIGHT_DIAM_CORR} / 染色あり÷{DARK_DIAM_CORR}")
    print(f"  染色なし設定      : blur=σ{BRIGHT_SIGMA}, prune={BRIGHT_PRUNE_MM}mm")
    print(f"  染色あり設定      : blur=σ{DARK_SIGMA},  prune={DARK_PRUNE_MM}mm")
    print(f"  根端フィルタ（明色根）: {BRIGHT_DIAM_LO}mm ≤ 直径 < {BRIGHT_DIAM_HI}mm")
    print(f"  根端フィルタ（暗色根）: 直径 < {DARK_DIAM_HI}mm")
    print(sep)

    rows   = []
    errors = []
    header = [
        "ファイル名", "解像度(DPI)", "根の種類", "Otsu閾値",
        "総根長(mm)", "根端数",
        "根表面積(cm2)", "根平均直径(mm)", "根の体積(cm3)",
    ]

    for i, img_path in enumerate(img_files, 1):
        print(f"\n[{i:2d}/{len(img_files)}] {img_path.name}")
        print(f"         解析中...", end="", flush=True)

        res = analyze_image(img_path)

        if res["error"]:
            print(f"\r         [エラー] {res['error']}")
            errors.append(img_path.name)
            rows.append([img_path.name] + [""] * 3 + ["ERROR"] * 5)
        else:
            polar_ja = "暗色根（染色あり）" if res["polarity"] == "dark" else "明色根（染色なし）"
            print(f"\r         完了               ")
            print(f"         種類       : {polar_ja}  DPI={res['dpi']:.0f}  Otsu={res['otsu_threshold']}")
            print(f"         総根長     : {res['total_length_mm']:.1f} mm")
            print(f"         根端数     : {res['tip_count']}")
            print(f"         根表面積   : {res['surface_cm2']:.4f} cm²")
            print(f"         根平均直径 : {res['mean_diam_mm']:.4f} mm")
            print(f"         根の体積   : {res['volume_cm3']:.4f} cm³")
            rows.append([
                res["filename"],
                f"{res['dpi']:.0f}",
                polar_ja,
                res["otsu_threshold"],
                res["total_length_mm"],
                res["tip_count"],
                res["surface_cm2"],
                res["mean_diam_mm"],
                res["volume_cm3"],
            ])

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n{sep}")
    print(f"  結果を保存しました : {output_csv}")
    print(f"  処理完了           : {len(img_files) - len(errors)}/{len(img_files)} 枚")
    if errors:
        print(f"  エラー ({len(errors)} 枚)      : {', '.join(errors)}")
    print(sep)


if __name__ == "__main__":
    main()

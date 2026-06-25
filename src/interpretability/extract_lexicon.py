import os
import hydra
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from omegaconf import DictConfig

# Đảm bảo import đúng DataModule từ source code của bạn
from src.data.data import EEGDataModule 

SFREQ = 256  # Tần số lấy mẫu của BCIC2A

# def extract_features_from_batch(eeg_batch, sfreq=SFREQ):
#     """
#     Trích xuất các đặc trưng thủ công từ một batch tín hiệu EEG.
#     """
#     if torch.is_tensor(eeg_batch):
#         eeg_batch = eeg_batch.detach().cpu().numpy()
        
#     batch_size, n_channels, n_times = eeg_batch.shape
#     per_channel_list = []
#     metadata_list = []
    
#     # 1. Nhóm Tần số (5 dải tần)
#     bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 45)}
#     freqs, psd = welch(eeg_batch, fs=sfreq, axis=-1, nperseg=min(n_times, 256))
#     for band_name, (fmin, fmax) in bands.items():
#         idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
#         band_power = psd[:, :, idx_band].mean(axis=-1) 
#         per_channel_list.append(band_power)
#         for ch in range(n_channels):
#             metadata_list.append({'Feature_Name': f'{band_name}_Power', 'Channel': ch + 1, 'Family': 'Frequency'})

#     # 2. Nhóm Thời gian (Thống kê)
#     mean_feat = eeg_batch.mean(axis=-1)
#     std_feat = eeg_batch.std(axis=-1)
#     skew_feat = skew(eeg_batch, axis=-1)
#     kurt_feat = kurtosis(eeg_batch, axis=-1)
#     for feat_arr, feat_name in zip([mean_feat, std_feat, skew_feat, kurt_feat], ['Mean', 'Std', 'Skew', 'Kurtosis']):
#         per_channel_list.append(feat_arr)
#         for ch in range(n_channels):
#             metadata_list.append({'Feature_Name': feat_name, 'Channel': ch + 1, 'Family': 'Time'})

#     # 3. Chỉ số Hjorth
#     hjorth_activity = std_feat ** 2
#     per_channel_list.append(hjorth_activity)
#     for ch in range(n_channels):
#         metadata_list.append({'Feature_Name': 'Hjorth_Activity', 'Channel': ch + 1, 'Family': 'Time'})
        
#     diff1 = np.diff(eeg_batch, axis=-1)
#     std_diff1 = diff1.std(axis=-1)
#     hjorth_mobility = std_diff1 / (std_feat + 1e-8)
#     per_channel_list.append(hjorth_mobility)
#     for ch in range(n_channels):
#         metadata_list.append({'Feature_Name': 'Hjorth_Mobility', 'Channel': ch + 1, 'Family': 'Time'})
        
#     diff2 = np.diff(diff1, axis=-1)
#     std_diff2 = diff2.std(axis=-1)
#     hjorth_complexity = (std_diff2 / (std_diff1 + 1e-8)) / (hjorth_mobility + 1e-8)
#     per_channel_list.append(hjorth_complexity)
#     for ch in range(n_channels):
#         metadata_list.append({'Feature_Name': 'Hjorth_Complexity', 'Channel': ch + 1, 'Family': 'Time'})

#     Z_per_channel = np.hstack(per_channel_list)

#     # 4. Đặc trưng toàn cục (Cross-channel)
#     global_features = []
#     for i in range(batch_size):
#         sample_data = eeg_batch[i]
#         corr_matrix = np.corrcoef(sample_data)
#         upper_tri_idx = np.triu_indices(n_channels, k=1)
#         corr_values = np.abs(corr_matrix[upper_tri_idx])
        
#         r001_mean = np.mean(corr_values)
#         r002_std = np.std(corr_values)
        
#         eigenvalues = np.linalg.eigvalsh(corr_matrix)
#         eigenvalues = np.maximum(eigenvalues, 1e-8)
#         norm_eigenvalues = eigenvalues / np.sum(eigenvalues)
#         r003_entropy = -np.sum(norm_eigenvalues * np.log(norm_eigenvalues + 1e-8))
#         r004_pr = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
        
#         global_features.append([r001_mean, r002_std, r003_entropy, r004_pr])
        
#     Z_global = np.array(global_features)
#     global_names = ['Global_Corr_Mean', 'Global_Corr_Std', 'Global_Eigen_Entropy', 'Global_Participation_Ratio']
#     for g_name in global_names:
#         metadata_list.append({'Feature_Name': g_name, 'Channel': 'All', 'Family': 'Cross-channel'})
        
#     Z_batch = np.hstack([Z_per_channel, Z_global])
#     return Z_batch, metadata_list

"""
eeg_features.py
================
Trich xuat lexicon 63 dac trung EEG thu cong theo Appendix A cua paper
"What Do EEG Foundation Models Capture from Human Brain Signals?"
(arXiv:2605.11410).

- 49 dac trung per-channel: Family T (10), F (16), TF (11), C (7), X (5)
- 14 dac trung global: Family R

Khac voi ban Appendix C.3 (328-cot, 5/6-statistic aggregation), o day moi
dac trung per-channel duoc giu nguyen 1 gia tri / channel, va moi dac trung
global la 1 gia tri / sample -> tong so cot = 49 * n_channels + 14.

Yeu cau:
    pip install antropy nolds --break-system-packages
"""
import sys
from importlib import _common

# Khởi tạo bản vá lỗi cho importlib để xử lý chuỗi module thay vì package
_orig_get_package = _common.get_package

def _patched_get_package(package):
    # Nếu nolds.datasets truyền vào dạng chuỗi, trả về package cha của nó là 'nolds'
    if package == 'nolds.datasets':
        return _orig_get_package('nolds')
    return _orig_get_package(package)

_common.get_package = _patched_get_package

import numpy as np
import torch
from scipy.signal import hilbert
from scipy.stats import kurtosis as _scipy_kurtosis
import antropy as ant
import nolds

EPS = 1e-12

# Canonical bands theo dung notation Appendix A (khac voi (1,4) cua ban cu cho Delta)
BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta":  (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}
BAND_NAMES = list(BANDS.keys())

N_PAC_BINS = 18      # so bin pha cho Tort modulation index
ACF_TAU_MAX = 100    # "implementation-fixed maximum lag" cho C006/C007


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bandpass_fft(x, fs, fmin, fmax):
    """FFT-domain rectangular bandpass. x: (..., T)."""
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= fmin) & (freqs < fmax)
    return np.fft.irfft(X * mask, n=n, axis=-1)


def _haar_dwt(x, levels=5):
    """Recursive Haar-style dyadic decomposition (even/odd +- averaging, 1/sqrt2 norm)."""
    details = []
    approx = np.asarray(x, dtype=np.float64)
    for _ in range(levels):
        n = approx.shape[-1]
        if n % 2 != 0:
            approx = approx[:-1]
            n -= 1
        if n < 2:
            details.append(np.zeros(1))
            continue
        even, odd = approx[0::2], approx[1::2]
        d = (even - odd) / np.sqrt(2.0)
        a = (even + odd) / np.sqrt(2.0)
        details.append(d)
        approx = a
    return details, approx


# --------------------------------------------------------------------------- #
# Family T: time-domain morphology (10, per-channel)
# --------------------------------------------------------------------------- #
def _family_T(x):
    diff1 = np.diff(x)
    diff2 = np.diff(diff1)
    var_x = np.var(x)
    var_d1 = np.var(diff1)
    var_d2 = np.var(diff2)

    t001_activity = var_x
    mobility = np.sqrt(var_d1 / (var_x + EPS))           # T002
    mobility_d1 = np.sqrt(var_d2 / (var_d1 + EPS))
    t003_complexity = mobility_d1 / (mobility + EPS)
    t004_std = np.sqrt(var_x)
    t005_rms = np.sqrt(np.mean(x ** 2))
    t006_kurtosis = _scipy_kurtosis(x, fisher=False, bias=False)
    t007_zcr = np.mean((x[:-1] * x[1:]) < 0)
    t008_line_length = np.mean(np.abs(diff1))
    t009_diff_std = np.std(diff1)
    t010_p2p = x.max() - x.min()

    return np.array([
        t001_activity, mobility, t003_complexity, t004_std, t005_rms,
        t006_kurtosis, t007_zcr, t008_line_length, t009_diff_std, t010_p2p,
    ])


# --------------------------------------------------------------------------- #
# Family F: spectral power and shape (16, per-channel)
# --------------------------------------------------------------------------- #
def _family_F(x, fs):
    n = len(x)
    X = np.fft.rfft(x)
    S = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    band_power, band_idx = {}, {}
    for name, (fmin, fmax) in BANDS.items():
        idx = (freqs >= fmin) & (freqs < fmax)
        band_idx[name] = idx
        band_power[name] = S[idx].sum()

    p_total = sum(band_power.values()) + EPS

    log_power = np.array([np.log(band_power[b] + EPS) for b in BAND_NAMES])     # F001-F005
    rel_power = np.array([band_power[b] / p_total for b in BAND_NAMES])         # F006-F010

    f011 = np.log((band_power["Theta"] + EPS) / (band_power["Beta"] + EPS))
    f012 = np.log((band_power["Delta"] + EPS) / (band_power["Alpha"] + EPS))
    f013 = np.log((band_power["Theta"] + EPS) / (band_power["Alpha"] + EPS))

    union_idx = np.zeros_like(freqs, dtype=bool)
    for idx in band_idx.values():
        union_idx |= idx
    f_union = freqs[union_idx]
    s_union = S[union_idx]
    n_f = max(len(f_union), 1)
    p_f = s_union / p_total

    f014_entropy = -np.sum(p_f * np.log(p_f + EPS)) / np.log(n_f + EPS)
    f015_centroid = np.sum(f_union * p_f)
    cumsum = np.cumsum(p_f)
    above = np.where(cumsum >= 0.95)[0]
    f016_edge95 = f_union[above[0]] if len(above) > 0 else f_union[-1]

    return np.concatenate([
        log_power, rel_power,
        [f011, f012, f013, f014_entropy, f015_centroid, f016_edge95],
    ])


# --------------------------------------------------------------------------- #
# Family TF: time-frequency envelope dynamics (11, per-channel)
# --------------------------------------------------------------------------- #
def _family_TF(x, fs):
    details, approx = _haar_dwt(x, levels=5)
    energies = [np.mean(d ** 2) for d in details] + [np.mean(approx ** 2)]
    e_total = sum(energies) + EPS
    p = np.array([e / e_total for e in energies])
    tf001_entropy = -np.sum(p * np.log(p + EPS)) / np.log(6)

    detail_vars = np.array([np.var(d) for d in details])  # TF002-TF006

    env_cv = []  # TF007-TF011
    for name, (fmin, fmax) in BANDS.items():
        xb = _bandpass_fft(x, fs, fmin, fmax)
        a = np.abs(hilbert(xb))
        env_cv.append(np.std(a) / (np.mean(a) + EPS))

    return np.concatenate([[tf001_entropy], detail_vars, env_cv])


# --------------------------------------------------------------------------- #
# Family C: signal complexity (7, per-channel) - antropy / nolds
# --------------------------------------------------------------------------- #
def _acf_decay_lags(x, tau_max=ACF_TAU_MAX):
    x = x - x.mean()
    if np.allclose(x, 0):
        return 1.0, 1.0

    acf_full = np.correlate(x, x, mode="full")
    acf = acf_full[len(acf_full) // 2:]
    acf = acf / (acf[0] + EPS)

    tau_max = max(min(int(tau_max), len(acf) - 1), 1)
    acf = acf[: tau_max + 1]

    below_1e = np.where(acf[1:] <= 1.0 / np.e)[0]
    tau_1e = (below_1e[0] + 1) if len(below_1e) > 0 else tau_max

    below_0 = np.where(acf[1:] <= 0.0)[0]
    tau_0 = (below_0[0] + 1) if len(below_0) > 0 else tau_max

    return tau_1e / tau_max, tau_0 / tau_max


def _family_C(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # C001: normalized order-3 permutation entropy
    try:
        c001 = ant.perm_entropy(x, order=3, delay=1, normalize=True)
    except Exception:
        c001 = np.nan

    # C002: sample entropy (order=2, chebyshev)
    try:
        c002 = ant.sample_entropy(x, order=2, metric="chebyshev")
        if not np.isfinite(c002):
            c002 = np.nan
    except Exception:
        c002 = np.nan

    # C003: Lempel-Ziv complexity on median-binarized sequence
    try:
        binary_seq = (x > np.median(x)).astype(int)
        c003 = ant.lziv_complexity(binary_seq, normalize=True)
    except Exception:
        c003 = np.nan

    # C004: Higuchi fractal dimension
    try:
        kmax = max(2, min(10, n // 4))
        c004 = ant.higuchi_fd(x, kmax=kmax)
    except Exception:
        c004 = np.nan

    # C005: DFA exponent
    try:
        c005 = nolds.dfa(x)
        if not np.isfinite(c005):
            c005 = np.nan
    except Exception:
        c005 = np.nan

    # C006 / C007: ACF-based decay lags
    c006, c007 = _acf_decay_lags(x)

    return np.array([c001, c002, c003, c004, c005, c006, c007])


# --------------------------------------------------------------------------- #
# Family X: cross-frequency coupling (5, per-channel)
# --------------------------------------------------------------------------- #
def _family_X(x, fs):
    def tort_mi(band_p, band_q):
        xp = _bandpass_fft(x, fs, *band_p)
        xq = _bandpass_fft(x, fs, *band_q)
        phase_p = np.angle(hilbert(xp))
        amp_q = np.abs(hilbert(xq))

        bin_edges = np.linspace(-np.pi, np.pi, N_PAC_BINS + 1)
        bin_idx = np.clip(np.digitize(phase_p, bin_edges) - 1, 0, N_PAC_BINS - 1)

        mean_amp = np.zeros(N_PAC_BINS)
        for j in range(N_PAC_BINS):
            mask = bin_idx == j
            mean_amp[j] = amp_q[mask].mean() if mask.any() else 0.0

        p = mean_amp / (mean_amp.sum() + EPS)
        kl = np.sum(p * np.log((p + EPS) * N_PAC_BINS))
        return kl / np.log(N_PAC_BINS)

    def aac(band_p, band_q):
        xp = _bandpass_fft(x, fs, *band_p)
        xq = _bandpass_fft(x, fs, *band_q)
        ap, aq = np.abs(hilbert(xp)), np.abs(hilbert(xq))
        if ap.std() < EPS or aq.std() < EPS:
            return 0.0
        r = np.corrcoef(ap, aq)[0, 1]
        return float(np.clip(r, -1.0, 1.0))

    x001 = tort_mi(BANDS["Delta"], BANDS["Beta"])    # PAC delta-beta
    x002 = tort_mi(BANDS["Theta"], BANDS["Gamma"])   # PAC theta-gamma
    x003 = tort_mi(BANDS["Alpha"], BANDS["Gamma"])   # PAC alpha-gamma
    x004 = aac(BANDS["Theta"], BANDS["Gamma"])       # AAC theta-gamma
    x005 = aac(BANDS["Delta"], BANDS["Gamma"])       # AAC delta-gamma

    return np.array([x001, x002, x003, x004, x005])


def _per_channel_features(x, fs):
    """49 = 10 (T) + 16 (F) + 11 (TF) + 7 (C) + 5 (X)."""
    return np.concatenate([
        _family_T(x),
        _family_F(x, fs),
        _family_TF(x, fs),
        _family_C(x),
        _family_X(x, fs),
    ])


# --------------------------------------------------------------------------- #
# Family R: cross-channel relations (14, global)
# --------------------------------------------------------------------------- #
def _family_R(sample_data, fs):
    """sample_data: (n_channels, T)."""
    n_channels = sample_data.shape[0]
    if n_channels < 2:
        return np.full(14, np.nan)

    upper_idx = np.triu_indices(n_channels, k=1)

    corr_matrix = np.corrcoef(sample_data)
    abs_corr = np.abs(corr_matrix[upper_idx])
    r001 = abs_corr.mean()
    r002 = abs_corr.std()

    # R003/R004 dung eigenvalues cua ma tran tuong quan CO DAU (khong phai |corr|)
    eigvals = np.linalg.eigvalsh(corr_matrix)
    eigvals = np.maximum(eigvals, EPS)
    eigvals = eigvals / eigvals.sum()
    r003_entropy = -np.sum(eigvals * np.log(eigvals + EPS)) / np.log(n_channels)
    r004_participation = 1.0 / np.sum(eigvals ** 2)

    Xf = np.fft.rfft(sample_data, axis=-1)
    freqs = np.fft.rfftfreq(sample_data.shape[-1], d=1.0 / fs)

    coherence, pli = [], []  # R005-R009, R010-R014
    for fmin, fmax in BANDS.values():
        idx = (freqs >= fmin) & (freqs < fmax)
        if idx.sum() == 0:
            coherence.append(0.0)
            pli.append(0.0)
            continue

        Xb = Xf[:, idx]
        Sxy = (Xb @ Xb.conj().T) / idx.sum()
        Sxx = np.real(np.diag(Sxy))
        denom = np.sqrt(np.outer(Sxx, Sxx)) + EPS
        coh = np.clip(np.abs(Sxy) / denom, 0.0, 1.0)
        coherence.append(coh[upper_idx].mean())

        xb = _bandpass_fft(sample_data, fs, fmin, fmax)
        phase = np.angle(hilbert(xb, axis=-1))
        pli_vals = [
            np.abs(np.mean(np.sign(np.sin(phase[i] - phase[j]))))
            for i, j in zip(*upper_idx)
        ]
        pli.append(np.mean(pli_vals))

    return np.concatenate([[r001, r002, r003_entropy, r004_participation], coherence, pli])


# --------------------------------------------------------------------------- #
# Feature registry / metadata
# --------------------------------------------------------------------------- #
FEATURE_REGISTRY = (
    [(f"T{i:03d}", name, "T") for i, name in enumerate([
        "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity", "Std",
        "RMS", "Kurtosis", "Zero_Crossing_Rate", "Line_Length",
        "Derivative_Std", "Peak_to_Peak",
    ], start=1)]
    + [(f"F{i:03d}", name, "F") for i, name in enumerate(
        [f"LogPower_{b}" for b in BAND_NAMES]
        + [f"RelPower_{b}" for b in BAND_NAMES]
        + ["LogRatio_Theta_Beta", "LogRatio_Delta_Alpha", "LogRatio_Theta_Alpha",
           "Spectral_Entropy", "Spectral_Centroid", "Spectral_Edge95"],
        start=1)]
    + [(f"TF{i:03d}", name, "TF") for i, name in enumerate(
        ["Wavelet_Subband_Entropy"]
        + [f"Wavelet_Detail_Var_L{k}" for k in range(1, 6)]
        + [f"Envelope_CV_{b}" for b in BAND_NAMES],
        start=1)]
    + [(f"C{i:03d}", name, "C") for i, name in enumerate([
        "Permutation_Entropy", "Sample_Entropy", "LZ_Complexity",
        "Higuchi_FD", "DFA_Exponent", "ACF_Decay_1e", "ACF_First_Zero",
    ], start=1)]
    + [(f"X{i:03d}", name, "X") for i, name in enumerate([
        "PAC_Delta_Beta", "PAC_Theta_Gamma", "PAC_Alpha_Gamma",
        "AAC_Theta_Gamma", "AAC_Delta_Gamma",
    ], start=1)]
)  # 49 entries

GLOBAL_FEATURE_REGISTRY = [
    (f"R{i:03d}", name, "R") for i, name in enumerate(
        ["Corr_Abs_Mean", "Corr_Abs_Std", "Eigen_Entropy", "Participation_Ratio"]
        + [f"Coherence_{b}" for b in BAND_NAMES]
        + [f"PLI_{b}" for b in BAND_NAMES],
        start=1)
]  # 14 entries


def get_feature_metadata(n_channels):
    """Metadata theo dung thu tu cot cua extract_features_from_batch."""
    metadata = []
    for feat_id, name, family in FEATURE_REGISTRY:
        for ch in range(n_channels):
            metadata.append({
                "Feature_ID": f"{feat_id}_ch{ch + 1}",
                "Feature_Name": name,
                "Channel": ch + 1,
                "Family": family,
            })
    for feat_id, name, family in GLOBAL_FEATURE_REGISTRY:
        metadata.append({
            "Feature_ID": feat_id,
            "Feature_Name": name,
            "Channel": "All",
            "Family": family,
        })
    return metadata


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def extract_features_from_batch(eeg_batch, sfreq=256):
    """
    Trich xuat 63 dac trung EEG (49 per-channel + 14 global) theo Appendix A.

    Parameters
    ----------
    eeg_batch : array-like hoac torch.Tensor, shape (batch_size, n_channels, n_times)
    sfreq : sampling rate (Hz)

    Returns
    -------
    Z_batch  : np.ndarray, shape (batch_size, 49 * n_channels + 14)
    metadata : list[dict]  (Feature_ID, Feature_Name, Channel, Family)
    """
    if torch.is_tensor(eeg_batch):
        eeg_batch = eeg_batch.detach().cpu().numpy()
    eeg_batch = np.asarray(eeg_batch, dtype=np.float64)

    batch_size, n_channels, _ = eeg_batch.shape
    metadata = get_feature_metadata(n_channels)

    rows = []
    for i in range(batch_size):
        sample = eeg_batch[i]  # (n_channels, T)

        per_channel = np.stack(
            [_per_channel_features(sample[c], sfreq) for c in range(n_channels)],
            axis=0,
        )  # (n_channels, 49)

        # feature-major flatten: [feat1_ch1..feat1_chN, feat2_ch1..feat2_chN, ...]
        per_channel_flat = per_channel.T.reshape(-1)

        global_feats = _family_R(sample, sfreq)

        rows.append(np.concatenate([per_channel_flat, global_feats]))

    Z_batch = np.vstack(rows)
    return Z_batch, metadata


# --- ĐÂY LÀ HÀM CHÍNH TÍCH HỢP ĐOẠN KHỞI TẠO CỦA BẠN VÀ HYDRA ---

@hydra.main(version_base=None, config_path="../../configs", config_name="example_config_downstream")
def main(cfg: DictConfig):
    # 0. Gọi đường dẫn đầu ra tự động từ Hydra (Giống code của bạn)
    from hydra.core.hydra_config import HydraConfig
    hydra_dir = HydraConfig.get().runtime.output_dir
    print(f"Hydra output directory: {hydra_dir}")
    
    # Định nghĩa thư mục lưu trữ các file đặc trưng trích xuất (.npy)
    output_dir = os.path.join(hydra_dir, "extracted_features")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Khởi tạo Data từ câu lệnh của bạn
    print("--- Đang khởi tạo DataModule ---")
    datamodule = EEGDataModule(cfg)
    datamodule.setup()
    
    # Tạo một DataLoader train cố định KHÔNG shuffle tự động (Bắt buộc)
    train_loader_fixed = torch.utils.data.DataLoader(
        datamodule.train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False, 
        num_workers=cfg.data.num_workers,
        persistent_workers=True
    )
    
    loaders = {
        "train": train_loader_fixed,
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader()
    }
    
    metadata_saved = False
    
    # 2. Tiến hành duyệt qua các loader và lưu file
    for split_name, loader in loaders.items():
        print(f"\n--- Tiến hành trích xuất cho tập: [{split_name.upper()}] ---")
        all_features = []
        all_labels = []
        
        for batch in tqdm(loader):
            # Xử lý linh hoạt cấu trúc batch dạng list/tuple (X, y) hoặc dict
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x = batch.get('eeg') or batch.get('data') or list(batch.values())[0]
                y = batch.get('label') or batch.get('target') or list(batch.values())[1]
            else:
                raise ValueError("Cấu trúc của batch không khớp!")
            
            # Tính toán ma trận đặc trưng
            Z_batch, meta = extract_features_from_batch(x, sfreq=SFREQ)
            
            all_features.append(Z_batch)
            if torch.is_tensor(y):
                y = y.detach().cpu().numpy()
            all_labels.append(y)
            
            # Lưu file mapping tên đặc trưng một lần duy nhất
            if not metadata_saved:
                df_meta = pd.DataFrame(meta)
                df_meta.insert(0, 'Feature_ID', range(len(meta)))
                df_meta.to_csv(os.path.join(output_dir, "feature_names.csv"), index=False)
                metadata_saved = True
                print(f"-> Đã tạo file quản lý tên đặc trưng: feature_names.csv (Tổng số cột: {len(meta)})")
                
        # Gom cụm lưu thành file npy
        Z_split = np.vstack(all_features)
        Y_split = np.concatenate(all_labels)
        
        np.save(os.path.join(output_dir, f"Z_{split_name}.npy"), Z_split)
        np.save(os.path.join(output_dir, f"Y_{split_name}.npy"), Y_split)
        print(f"-> Lưu thành công {split_name}: Z_shape={Z_split.shape}, Y_shape={Y_split.shape}")

if __name__ == "__main__":
    main()
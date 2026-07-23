# import matplotlib.pyplot as plt
# import pandas as pd


# def load_and_filter_loss(file_path, metric_name):
#     """Đọc file CSV và lọc ra dữ liệu của metric cần thiết, sắp xếp theo Step."""
#     try:
#         df = pd.read_csv(file_path)
#         filtered_df = df[df["Metric"] == metric_name].sort_values(by="Step")
#         return filtered_df
#     except FileNotFoundError:
#         print(f"Cảnh báo: Không tìm thấy file {file_path}. Bỏ qua file này.")
#         return None
#     except Exception as e:
#         print(f"Lỗi khi đọc file {file_path}: {e}")
#         return None


# # --- 1. Cấu hình đường dẫn file và Metric ---
# # Thay đổi đường dẫn chính xác tới các file của bạn nếu cần
# files_reconstruction = {
#     "FAMO": "cbramod_famo.csv",
#     "OGR": "cbramod_ogr.csv",
#     "Linear": "cbramod_linear.csv",
#     "Reconstruction Only": "cbramod_reconstruction.csv",
# }

# files_contrastive = {
#     "FAMO": "cbramod_famo.csv",
#     "OGR": "cbramod_ogr.csv",
#     "Linear": "cbramod_linear.csv",
#     "Contrastive Only": "cbramod_contrastive.csv",
# }

# metric_recon = "train/reconstruction_loss"
# metric_contrastive = "train/byol_loss"

# # --- 2. Khởi tạo Figure gồm 2 subplots (1 hàng, 2 cột) ---
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # --- 3. Vẽ biểu đồ Reconstruction Loss ---
# print("Đang xử lý biểu đồ Reconstruction Loss...")
# for label, path in files_reconstruction.items():
#     data = load_and_filter_loss(path, metric_recon)
#     if data is not None and not data.empty:
#         ax1.plot(data["Step"], data["Value"], label=label, linewidth=1.5)

# ax1.set_title("Reconstruction Loss Comparison", fontsize=14, fontweight="bold")
# ax1.set_xlabel("Step", fontsize=12)
# ax1.set_ylabel("Loss Value", fontsize=12)
# ax1.grid(True, linestyle="--", alpha=0.6)
# ax1.legend(fontsize=11)

# # --- 4. Vẽ biểu đồ Contrastive Loss ---
# print("Đang xử lý biểu đồ Contrastive Loss...")
# for label, path in files_contrastive.items():
#     data = load_and_filter_loss(path, metric_contrastive)
#     if data is not None and not data.empty:
#         ax2.plot(data["Step"], data["Value"], label=label, linewidth=1.5)

# ax2.set_title("Contrastive (BYOL) Loss Comparison", fontsize=14, fontweight="bold")
# ax2.set_xlabel("Step", fontsize=12)
# ax2.set_ylabel("Loss Value", fontsize=12)
# ax2.grid(True, linestyle="--", alpha=0.6)
# ax2.legend(fontsize=11)

# # --- 5. Tối ưu hiển thị và lưu/hiện biểu đồ ---
# plt.tight_layout()
# plt.savefig("loss_comparison_plot.png", dpi=300)  # Lưu lại thành file ảnh
# plt.show()

##########################################3

import matplotlib.pyplot as plt
import pandas as pd


def load_and_filter_loss(file_path, metric_name):
    """Đọc file CSV và lọc ra dữ liệu của metric cần thiết, sắp xếp theo Step."""
    try:
        df = pd.read_csv(file_path)
        filtered_df = df[df["Metric"] == metric_name].sort_values(by="Step")
        return filtered_df
    except FileNotFoundError:
        print(f"Cảnh báo: Không tìm thấy file {file_path}. Bỏ qua file này.")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        return None


# --- 1. Cấu hình đường dẫn file và Metric ---
# Thay đổi đường dẫn chính xác tới các file của bạn nếu cần
files_reconstruction = {
    "FAMO": "cbramod_famo.csv",
    "OGR": "cbramod_ogr.csv",
    "Linear": "cbramod_linear.csv",
    "Reconstruction Only": "cbramod_reconstruction.csv",
}

files_contrastive = {
    "FAMO": "cbramod_famo.csv",
    "OGR": "cbramod_ogr.csv",
    "Linear": "cbramod_linear.csv",
    "Contrastive Only": "cbramod_contrastive.csv",
}

metric_recon = "train/reconstruction_loss"
metric_contrastive = "train/byol_reg_loss"

# --- 2. Khởi tạo Figure gồm 2 subplots (1 hàng, 2 cột) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- 3. Vẽ biểu đồ Reconstruction Loss ---
print("Đang xử lý biểu đồ Reconstruction Loss...")
for label, path in files_reconstruction.items():
    data = load_and_filter_loss(path, metric_recon)
    if data is not None and not data.empty:
        ax1.plot(data["Step"], data["Value"], label=label, linewidth=1.5)

ax1.set_title("Reconstruction Loss Comparison", fontsize=14, fontweight="bold")
ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("Loss Value", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=11)

# --- 4. Vẽ biểu đồ Contrastive Loss ---
print("Đang xử lý biểu đồ Contrastive Loss...")
for label, path in files_contrastive.items():
    data = load_and_filter_loss(path, metric_contrastive)
    if data is not None and not data.empty:
        ax2.plot(data["Step"], data["Value"], label=label, linewidth=1.5)

ax2.set_title("Contrastive (BYOL) Loss Comparison", fontsize=14, fontweight="bold")
ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("Loss Value", fontsize=12)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=11)

# --- 5. Tối ưu hiển thị và lưu/hiện biểu đồ ---
plt.tight_layout()
plt.savefig("cbramod_loss_comparison_plot.png", dpi=300)  # Lưu lại thành file ảnh
plt.show()
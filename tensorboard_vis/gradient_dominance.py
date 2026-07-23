# import pandas as pd
# import matplotlib.pyplot as plt

# def process_file(filepath):
#     # 1. Đọc dữ liệu từ file csv
#     df = pd.read_csv(filepath)
    
#     # 2. Lấy trực tiếp các metric cosine similarity đã được tính sẵn trong code của bạn
#     target_metrics = [
#         'gradient_monitor/grad/byol_reg_cos_total',
#         'gradient_monitor/grad/reconstruction_cos_total'
#     ]
    
#     # 3. Lọc df chỉ lấy các metric này
#     df_filtered = df[df['Metric'].isin(target_metrics)]
    
#     # 4. Xoay bảng dựa trên dữ liệu đã lọc
#     pivoted = df_filtered.pivot(index='Step', columns='Metric', values='Value')
    
#     # 5. Đổi tên cột cho ngắn gọn, dễ vẽ
#     pivoted = pivoted.rename(columns={
#         'gradient_monitor/grad/byol_reg_cos_total': 'byol_reg_cos',
#         'gradient_monitor/grad/reconstruction_cos_total': 'recon_cos'
#     })
    
#     # 6. Loại bỏ các dòng bị thiếu dữ liệu (nếu có) và reset index
#     return pivoted.dropna(subset=['byol_reg_cos', 'recon_cos']).reset_index()

# # 1. Đọc và xử lý các file dữ liệu (Bật/tắt comment tùy theo file bạn có)
# df_famo = process_file('cbramod_famo.csv')
# df_linear = process_file('cbramod_linear.csv')
# df_ogr = process_file('cbramod_ogr.csv')

# # 2. Khởi tạo khung vẽ gồm 2 biểu đồ con (subplots)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # --- BIỂU ĐỒ 1: BYOL Reg Gradient Cosine Similarity ---
# # Sửa ở đây: Sử dụng 'byol_reg_cos' thay vì 'byol_reg_ratio'
# ax1.plot(df_famo['Step'], df_famo['byol_reg_cos'], label='FAMO', color='red', linewidth=2)
# ax1.plot(df_linear['Step'], df_linear['byol_reg_cos'], label='Linear', color='blue', linestyle='--')
# ax1.plot(df_ogr['Step'], df_ogr['byol_reg_cos'], label='OGR', color='green', linestyle='-.')

# # Đường tham chiếu y = 0 để phân định rõ vùng Xung đột (<0) và Đồng thuận (>0)
# ax1.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# ax1.set_xlabel('Training Step', fontsize=11)
# ax1.set_ylabel('Cosine Similarity', fontsize=11)
# ax1.set_title('BYOL Reg Gradient Cosine Similarity', fontsize=13, fontweight='bold')
# ax1.set_ylim(-1.1, 1.1) # Cố định trục Y từ -1 đến 1
# ax1.grid(True, linestyle=':', alpha=0.6)
# ax1.legend(fontsize=10)

# # --- BIỂU ĐỒ 2: Reconstruction Gradient Cosine Similarity ---
# # Sửa ở đây: Sử dụng 'recon_cos' thay vì 'recon_ratio'
# ax2.plot(df_famo['Step'], df_famo['recon_cos'], label='FAMO', color='red', linewidth=2)
# ax2.plot(df_linear['Step'], df_linear['recon_cos'], label='Linear', color='blue', linestyle='--')
# ax2.plot(df_ogr['Step'], df_ogr['recon_cos'], label='OGR', color='green', linestyle='-.')

# # Đường tham chiếu y = 0
# ax2.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# ax2.set_xlabel('Training Step', fontsize=11)
# ax2.set_ylabel('Cosine Similarity', fontsize=11)
# ax2.set_title('Reconstruction Gradient Cosine Similarity', fontsize=13, fontweight='bold')
# ax2.set_ylim(-1.1, 1.1) # Cố định trục Y giống ax1
# ax2.grid(True, linestyle=':', alpha=0.6)
# ax2.legend(fontsize=10)

# # 3. Tối ưu hiển thị và lưu biểu đồ
# plt.tight_layout()
# plt.savefig('cbramod_gradient_cosines.png', dpi=300)
# plt.show()

######################################################

# import pandas as pd
# import matplotlib.pyplot as plt

# def process_file(filepath):
#     # 1. Đọc dữ liệu từ file csv
#     df = pd.read_csv(filepath)
    
#     # 2. Lấy trực tiếp các metric cosine similarity đã được tính sẵn trong code của bạn
#     target_metrics = [
#         'gradient_monitor/grad/byol_cos_total',
#         'gradient_monitor/grad/reconstruction_cos_total'
#     ]
    
#     # 3. Lọc df chỉ lấy các metric này
#     df_filtered = df[df['Metric'].isin(target_metrics)]
    
#     # 4. Xoay bảng dựa trên dữ liệu đã lọc
#     pivoted = df_filtered.pivot(index='Step', columns='Metric', values='Value')
    
#     # 5. Đổi tên cột cho ngắn gọn, dễ vẽ
#     pivoted = pivoted.rename(columns={
#         'gradient_monitor/grad/byol_cos_total': 'byol_cos',
#         'gradient_monitor/grad/reconstruction_cos_total': 'recon_cos'
#     })
    
#     # 6. Loại bỏ các dòng bị thiếu dữ liệu (nếu có) và reset index
#     return pivoted.dropna(subset=['byol_cos', 'recon_cos']).reset_index()

# # 1. Đọc và xử lý các file dữ liệu (Bật/tắt comment tùy theo file bạn có)
# df_famo = process_file('eegpt_famo.csv')
# df_linear = process_file('eegpt_linear.csv')
# df_ogr = process_file('eegpt_ogr.csv')

# # 2. Khởi tạo khung vẽ gồm 2 biểu đồ con (subplots)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # --- BIỂU ĐỒ 1: BYOL Reg Gradient Cosine Similarity ---
# # Sửa ở đây: Sử dụng 'byol_reg_cos' thay vì 'byol_reg_ratio'
# ax1.plot(df_famo['Step'], df_famo['byol_cos'], label='FAMO', color='red', linewidth=2)
# ax1.plot(df_linear['Step'], df_linear['byol_cos'], label='Linear', color='blue', linestyle='--')
# ax1.plot(df_ogr['Step'], df_ogr['byol_cos'], label='OGR', color='green', linestyle='-.')

# # Đường tham chiếu y = 0 để phân định rõ vùng Xung đột (<0) và Đồng thuận (>0)
# ax1.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# ax1.set_xlabel('Training Step', fontsize=11)
# ax1.set_ylabel('Cosine Similarity', fontsize=11)
# ax1.set_title('BYOL Reg Gradient Cosine Similarity', fontsize=13, fontweight='bold')
# ax1.set_ylim(-1.1, 1.1) # Cố định trục Y từ -1 đến 1
# ax1.grid(True, linestyle=':', alpha=0.6)
# ax1.legend(fontsize=10)

# # --- BIỂU ĐỒ 2: Reconstruction Gradient Cosine Similarity ---
# # Sửa ở đây: Sử dụng 'recon_cos' thay vì 'recon_ratio'
# ax2.plot(df_famo['Step'], df_famo['recon_cos'], label='FAMO', color='red', linewidth=2)
# ax2.plot(df_linear['Step'], df_linear['recon_cos'], label='Linear', color='blue', linestyle='--')
# ax2.plot(df_ogr['Step'], df_ogr['recon_cos'], label='OGR', color='green', linestyle='-.')

# # Đường tham chiếu y = 0
# ax2.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# ax2.set_xlabel('Training Step', fontsize=11)
# ax2.set_ylabel('Cosine Similarity', fontsize=11)
# ax2.set_title('Reconstruction Gradient Cosine Similarity', fontsize=13, fontweight='bold')
# ax2.set_ylim(-1.1, 1.1) # Cố định trục Y giống ax1
# ax2.grid(True, linestyle=':', alpha=0.6)
# ax2.legend(fontsize=10)

# # 3. Tối ưu hiển thị và lưu biểu đồ
# plt.tight_layout()
# plt.savefig('eegpt_gradient_cosines.png', dpi=300)
# plt.show()

######################################################3

# import pandas as pd
# import matplotlib.pyplot as plt

# def process_file(filepath):
#     # 1. Đọc dữ liệu từ file csv
#     df = pd.read_csv(filepath)
    
#     # 2. Cần thêm total_norm để nhân với cosine similarity
#     target_metrics = [
#         'gradient_monitor/grad/total_norm',
#         'gradient_monitor/grad/byol_reg_cos_total',
#         'gradient_monitor/grad/reconstruction_cos_total',
#         'gradient_monitor/grad/byol_reg_norm',
#         'gradient_monitor/grad/reconstruction_norm',
#     ]
    
#     # 3. Lọc df chỉ lấy các metric này
#     df_filtered = df[df['Metric'].isin(target_metrics)]
    
#     # 4. Xoay bảng dựa trên dữ liệu đã lọc
#     pivoted = df_filtered.pivot(index='Step', columns='Metric', values='Value')
    
#     # 5. Tính toán chỉ số mới: Cosine Sim * Total Norm
#     total_norm = pivoted['gradient_monitor/grad/total_norm']
#     total_norm = total_norm.clip(upper=1.0)
    
#     pivoted['byol_reg_cos_norm'] = pivoted['gradient_monitor/grad/byol_reg_cos_total'] * total_norm 
#     pivoted['recon_cos_norm'] = pivoted['gradient_monitor/grad/reconstruction_cos_total'] * total_norm 
#     # 6. Loại bỏ các dòng bị thiếu dữ liệu (nếu có) và reset index
#     return pivoted.dropna(subset=['byol_reg_cos_norm', 'recon_cos_norm']).reset_index()

# # 1. Đọc và xử lý các file dữ liệu
# df_famo = process_file('cbramod_famo.csv')
# df_linear = process_file('cbramod_linear.csv')
# df_ogr = process_file('cbramod_ogr.csv')

# # 2. Khởi tạo khung vẽ gồm 2 biểu đồ con (subplots)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # --- BIỂU ĐỒ 1: BYOL Reg Gradient Projection onto Task ---
# ax1.plot(df_famo['Step'], df_famo['byol_reg_cos_norm'], label='FAMO', color='red', linewidth=2)
# ax1.plot(df_linear['Step'], df_linear['byol_reg_cos_norm'], label='Linear', color='blue', linestyle='--')
# ax1.plot(df_ogr['Step'], df_ogr['byol_reg_cos_norm'], label='OGR', color='green', linestyle='-.')

# # Đường tham chiếu y = 0 để phân định rõ vùng Xung đột (<0) và Đồng thuận (>0)
# ax1.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# ax1.set_xlabel('Training Step', fontsize=11)
# ax1.set_ylabel('Proj (Total onto Task)', fontsize=11)
# ax1.set_title('BYOL Reg Gradient Projection (Cos * Total Norm)', fontsize=13, fontweight='bold')
# ax1.grid(True, linestyle=':', alpha=0.6)
# ax1.legend(fontsize=10)

# # ax1.set_yscale('symlog')

# # --- BIỂU ĐỒ 2: Reconstruction Gradient Projection onto Task ---
# ax2.plot(df_famo['Step'], df_famo['recon_cos_norm'], label='FAMO', color='red', linewidth=2)
# ax2.plot(df_linear['Step'], df_linear['recon_cos_norm'], label='Linear', color='blue', linestyle='--')
# ax2.plot(df_ogr['Step'], df_ogr['recon_cos_norm'], label='OGR', color='green', linestyle='-.')

# # Đường tham chiếu y = 0
# ax2.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# ax2.set_xlabel('Training Step', fontsize=11)
# ax2.set_ylabel('Proj (Total onto Task)', fontsize=11)
# ax2.set_title('Reconstruction Gradient Projection (Cos * Total Norm)', fontsize=13, fontweight='bold')
# ax2.grid(True, linestyle=':', alpha=0.6)
# ax2.legend(fontsize=10)

# # 3. Tối ưu hiển thị và lưu biểu đồ
# plt.tight_layout()
# plt.savefig('cbramod_gradient_cos_norm.png', dpi=300)
# plt.show()

######################################################3

# import pandas as pd
# import matplotlib.pyplot as plt

# def process_file(filepath):
#     # 1. Đọc dữ liệu từ file csv
#     df = pd.read_csv(filepath)
    
#     # 2. Cần thêm total_norm để nhân với cosine similarity
#     target_metrics = [
#         'gradient_monitor/grad/total_norm',
#         'gradient_monitor/grad/byol_cos_total',
#         'gradient_monitor/grad/reconstruction_cos_total',
#         'gradient_monitor/grad/byol_norm',
#         'gradient_monitor/grad/reconstruction_norm',
#     ]
    
#     # 3. Lọc df chỉ lấy các metric này
#     df_filtered = df[df['Metric'].isin(target_metrics)]
    
#     # 4. Xoay bảng dựa trên dữ liệu đã lọc
#     pivoted = df_filtered.pivot(index='Step', columns='Metric', values='Value')
    
#     # 5. Tính toán chỉ số mới: Cosine Sim * Total Norm
#     total_norm = pivoted['gradient_monitor/grad/total_norm']
#     total_norm = total_norm.clip(upper=1.0)
    
#     pivoted['byol_cos_norm'] = pivoted['gradient_monitor/grad/byol_cos_total'] * total_norm 
#     pivoted['recon_cos_norm'] = pivoted['gradient_monitor/grad/reconstruction_cos_total'] * total_norm 
#     # 6. Loại bỏ các dòng bị thiếu dữ liệu (nếu có) và reset index
#     return pivoted.dropna(subset=['byol_cos_norm', 'recon_cos_norm']).reset_index()

# # 1. Đọc và xử lý các file dữ liệu
# df_famo = process_file('eegpt_famo.csv')
# df_linear = process_file('eegpt_linear.csv')
# df_ogr = process_file('eegpt_ogr.csv')

# # 2. Khởi tạo khung vẽ gồm 2 biểu đồ con (subplots)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # --- BIỂU ĐỒ 1: BYOL Reg Gradient Projection onto Task ---
# ax1.plot(df_famo['Step'], df_famo['byol_cos_norm'], label='FAMO', color='red', linewidth=2)
# ax1.plot(df_linear['Step'], df_linear['byol_cos_norm'], label='Linear', color='blue', linestyle='--')
# ax1.plot(df_ogr['Step'], df_ogr['byol_cos_norm'], label='OGR', color='green', linestyle='-.')

# # Đường tham chiếu y = 0 để phân định rõ vùng Xung đột (<0) và Đồng thuận (>0)
# ax1.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# ax1.set_xlabel('Training Step', fontsize=11)
# ax1.set_ylabel('Proj (Total onto Task)', fontsize=11)
# ax1.set_title('BYOL Reg Gradient Projection (Cos * Total Norm)', fontsize=13, fontweight='bold')
# ax1.grid(True, linestyle=':', alpha=0.6)
# ax1.legend(fontsize=10)

# # ax1.set_yscale('symlog')

# # --- BIỂU ĐỒ 2: Reconstruction Gradient Projection onto Task ---
# ax2.plot(df_famo['Step'], df_famo['recon_cos_norm'], label='FAMO', color='red', linewidth=2)
# ax2.plot(df_linear['Step'], df_linear['recon_cos_norm'], label='Linear', color='blue', linestyle='--')
# ax2.plot(df_ogr['Step'], df_ogr['recon_cos_norm'], label='OGR', color='green', linestyle='-.')

# # Đường tham chiếu y = 0
# ax2.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# ax2.set_xlabel('Training Step', fontsize=11)
# ax2.set_ylabel('Proj (Total onto Task)', fontsize=11)
# ax2.set_title('Reconstruction Gradient Projection (Cos * Total Norm)', fontsize=13, fontweight='bold')
# ax2.grid(True, linestyle=':', alpha=0.6)
# ax2.legend(fontsize=10)

# # 3. Tối ưu hiển thị và lưu biểu đồ
# plt.tight_layout()
# plt.savefig('eegpt_gradient_cos_norm.png', dpi=300)
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def process_file(filepath):
    # 1. Đọc dữ liệu từ file csv
    df = pd.read_csv(filepath)
    
    # 2. Cần thêm total_norm để nhân với cosine similarity
    target_metrics = [
        'gradient_monitor/grad/total_norm',
        'gradient_monitor/grad/byol_cos_total',
        'gradient_monitor/grad/reconstruction_cos_total',
        'gradient_monitor/grad/byol_norm',
        'gradient_monitor/grad/reconstruction_norm',
    ]
    
    # 3. Lọc df chỉ lấy các metric này
    df_filtered = df[df['Metric'].isin(target_metrics)]
    
    # 4. Xoay bảng dựa trên dữ liệu đã lọc
    pivoted = df_filtered.pivot(index='Step', columns='Metric', values='Value')
    
    # 5. Tính toán chỉ số mới: Cosine Sim * Total Norm
    total_norm = pivoted['gradient_monitor/grad/total_norm']
    total_norm = total_norm.clip(upper=1.0)
    
    pivoted['byol_cos_norm'] = pivoted['gradient_monitor/grad/byol_cos_total'] * total_norm 
    pivoted['recon_cos_norm'] = pivoted['gradient_monitor/grad/reconstruction_cos_total'] * total_norm 
    
    # 6. Loại bỏ các dòng bị thiếu dữ liệu (nếu có) và reset index
    return pivoted.dropna(subset=['byol_cos_norm', 'recon_cos_norm']).reset_index()

# 1. Đọc và xử lý các file dữ liệu
df_famo = process_file('eegpt_famo.csv')
df_linear = process_file('eegpt_linear.csv')
df_ogr = process_file('eegpt_ogr.csv')

# 2. Khởi tạo khung vẽ SINGLE PLOT (1 biểu đồ duy nhất)
plt.figure(figsize=(10, 6))

# --- VẼ ĐƯỜNG CHO BYOL (Nét liền '-') ---
plt.plot(df_famo['Step'], df_famo['byol_cos_norm'], label='FAMO (BYOL)', color='red', linestyle='-', linewidth=2)
plt.plot(df_linear['Step'], df_linear['byol_cos_norm'], label='Linear (BYOL)', color='blue', linestyle='-', linewidth=1.5)
plt.plot(df_ogr['Step'], df_ogr['byol_cos_norm'], label='OGR (BYOL)', color='green', linestyle='-', linewidth=1.5)

# --- VẼ ĐƯỜNG CHO RECONSTRUCTION (Nét đứt '--') ---
plt.plot(df_famo['Step'], df_famo['recon_cos_norm'], label='FAMO (Recon)', color='red', linestyle='--', linewidth=2)
plt.plot(df_linear['Step'], df_linear['recon_cos_norm'], label='Linear (Recon)', color='blue', linestyle='--', linewidth=1.5)
plt.plot(df_ogr['Step'], df_ogr['recon_cos_norm'], label='OGR (Recon)', color='green', linestyle='--', linewidth=1.5)

# Đường tham chiếu y = 0 để phân định rõ vùng Xung đột (<0) và Đồng thuận (>0)
plt.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# Cấu hình các thông tin trục và tiêu đề
plt.xlabel('Training Step', fontsize=11)
plt.ylabel('Proj (Total onto Task)', fontsize=11)
plt.title('Gradient Projection Comparison (Cos * Total Norm)', fontsize=13, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)

# Đặt legend ở vị trí tối ưu để không che khuất đường vẽ
plt.legend(fontsize=10, loc='best')

# 3. Tối ưu hiển thị và lưu biểu đồ
plt.tight_layout()
plt.savefig('eegpt_gradient_cos_norm_combined.png', dpi=300)
plt.show()
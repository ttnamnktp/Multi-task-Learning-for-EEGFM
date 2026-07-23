import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def export_selected_tensorboard_metrics(log_dir, output_csv, target_tags=None):
    """
    Trích xuất CHỈ các biểu đồ được chỉ định từ TensorBoard sang CSV.
    :param log_dir: Thư mục chứa file log TensorBoard.
    :param output_csv: Tên file CSV xuất ra.
    :param target_tags: List chứa tên các biểu đồ muốn lấy (Ví dụ: ['train/loss', 'val/loss'])
                        Nếu để None hoặc trống, code sẽ tự động lấy HẾT.
    """
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()

    all_data = []
    available_tags = ea.Tags()['scalars']
    
    # Xác định danh sách tags sẽ lấy
    if target_tags:
        # Chỉ lấy những tag vừa nằm trong danh sách yêu cầu, vừa có thật trong log
        tags_to_extract = [tag for tag in target_tags if tag in available_tags]
        
        # Cảnh báo nếu bạn gõ nhầm tên tag nào đó không tồn tại
        missing_tags = set(target_tags) - set(available_tags)
        if missing_tags:
            print(f"⚠️ Cảnh báo: Không tìm thấy các tag này trong log: {list(missing_tags)}")
    else:
        tags_to_extract = available_tags

    # Tiến hành trích xuất
    for tag in tags_to_extract:
        events = ea.Scalars(tag)
        for event in events:
            all_data.append({
                'Metric': tag,
                'Step': event.step,
                'Value': event.value,
                'Wall_Time': event.wall_time
            })

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"✅ Đã xuất thành công {len(tags_to_extract)} biểu đồ được chọn vào file: {output_csv}")
    else:
        print("❌ Không trích xuất được dữ liệu nào. Vui lòng kiểm tra lại đường dẫn hoặc tên Metric.")

# --- CÁCH SỬ DỤNG ---
# log_directory = '/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/879352/2026-07-02/15-57-38/tb_logs/version_0' 
# output_file = 'cbramod_reconstruction.csv'

output_files = [
#     'eegpt_reconstruction.csv',
#     'eegpt_contrastive.csv',
#     'eegpt_linear.csv',
#     'eegpt_ogr.csv',
#     'eegpt_famo.csv',
# ]

# log_dirs = [
#     '/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/877240/2026-06-30/17-51-21/tb_logs/version_0',
#     '/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/877241/2026-06-30/18-14-36/tb_logs/version_0',
#     '/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/877244/2026-06-30/19-33-30/tb_logs/version_0',
#     '/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/853901/2026-06-17/10-22-49/tb_logs/version_0',
#     '/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/880534/2026-07-03/17-07-18/tb_logs/version_0',
# ]

    'cbramod_reconstruction.csv',
    'cbramod_contrastive.csv',
    'cbramod_linear.csv',
    'cbramod_ogr.csv',
    'cbramod_famo.csv',
]

log_dirs = [
    '/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/879352/2026-07-02/15-57-38/tb_logs/version_0',
    '/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/879351/2026-07-02/15-21-39/tb_logs/version_0',
    '/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/859555/2026-06-22/16-39-13/tb_logs/version_0',
    '/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/859558/2026-06-22/16-40-12/tb_logs/version_0',
    '/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/864959/2026-06-26/16-33-10/tb_logs/version_0',
]

my_favorite_metrics = [
    'gradient_monitor/grad/reconstruction_norm',
    'gradient_monitor/grad/reconstruction_proj_total',
    'gradient_monitor/grad/byol_reg_norm',
    'gradient_monitor/grad/byol_reg_proj_total',
    'gradient_monitor/grad/byol_reg_cos_total',
    'gradient_monitor/grad/reconstruction_cos_total',
    'gradient_monitor/grad/total_norm',
    'train/byol_reg_loss',
    'train/reconstruction_loss'
]

# my_favorite_metrics = [
#     'gradient_monitor/grad/reconstruction_norm',
#     'gradient_monitor/grad/reconstruction_proj_total',
#     'gradient_monitor/grad/byol_norm',
#     'gradient_monitor/grad/byol_proj_total',
#     'gradient_monitor/grad/byol_cos_total',
#     'gradient_monitor/grad/reconstruction_cos_total',
#     'gradient_monitor/grad/total_norm',
#     'train/byol_loss',
#     'train/reconstruction_loss',   
# ]

for i in range(0,5):
    export_selected_tensorboard_metrics(log_dirs[i], output_files[i], target_tags=my_favorite_metrics)
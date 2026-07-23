# test_datamodule.py
# USAGE: Enter this prompt
# PYTHONPATH=. python src/data/_test_data_module.py

from omegaconf import OmegaConf
from src.data.data import EEGDataModule 

def main():
    cfg = OmegaConf.create({
        # "dataset": {
        #     "name": "physiomi",
        #     "data_dir": "/projects/EEG-foundation-model/tran-24/data_200Hz_processed/physiomi_no_car",  # <-- đổi lại
        #     "seed": 7,
        #     "test_fold_index": 4,
        #     "cv_fold_index": 0,
        #     "folds_json": None,
        #     "scale_div": 1000.0,
        # },
        # "dataset": {
        #     "name": "kaggle_ern",
        #     "data_dir": "/projects/EEG-foundation-model/tran-24/data_200Hz_processed/KaggleERN_200hz_no_CAR",  # <-- đổi lại
        #     "seed": 7,
        #     "fold": 5,
        #     "scale_div": 1000.0,
        # },
        "dataset": {
            "name": "sleepedf",
            "data_dir": "/projects/EEG-foundation-model/tran-24/data_200Hz_processed/sleep_edf_bipolar_no_CAR",  # <-- đổi lại
            "seed": 7,
            "n_folds": 5,
            "fold": 0,
            "split_mode": "subject_kfold",
            "val_ratio": 0.2,
            "stratified_val": True,
            "scale_div": 200.0,
        },
        # "dataset": {
        #     "name": "tuab",
        #     "data_dir": "/projects/EEG-foundation-model/tran-24/data_200Hz_processed/tuab_no_car",  # <-- đổi lại
        #     "seed": 7,
        #     "val_subject_ids": ["aaaaagvr", "aaaaalej"],
        #     "scale_div": 100.0
        # },
        "data": {
            "batch_size": 32,
            "num_workers": 2,  # 0 để debug dễ hơn
        }
    })

    print("=" * 50)
    print("Khởi tạo DataModule...")
    dm = EEGDataModule(cfg)
    dm.setup()

    # ── Kiểm tra size ──────────────────────────────────
    print(f"\nSplit sizes:")
    print(f"  train : {len(dm.train_ds):>6} samples")
    print(f"  val   : {len(dm.val_ds):>6} samples")
    print(f"  test  : {len(dm.test_ds):>6} samples")

    # ── Kiểm tra 1 batch từ mỗi loader ────────────────
    for split, loader in [
        ("train", dm.train_dataloader()),
        ("val",   dm.val_dataloader()),
        ("test",  dm.test_dataloader()),
    ]:
        x, y = next(iter(loader))
        print(f"\n[{split}] batch:")
        print(f"  x shape : {x.shape}   dtype={x.dtype}")
        print(f"  y shape : {y.shape}   dtype={y.dtype}")
        print(f"  x range : [{x.min():.3f}, {x.max():.3f}]")
        print(f"  labels  : {y[:32].tolist()}")

    print("\n✓ DataModule load thành công!")

if __name__ == "__main__":
    main()
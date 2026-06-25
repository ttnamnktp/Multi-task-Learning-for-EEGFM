from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

def build_callbacks(cfg):
    ckpt = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-{epoch}-{val_acc:.4f}",
        verbose=True,
        save_last=True
    )

    # early_stop = EarlyStopping(
    #     monitor="val_acc",
    #     mode="max",
    #     patience=10
    # )

    # return [ckpt, early_stop], ckpt
    return [ckpt], ckpt
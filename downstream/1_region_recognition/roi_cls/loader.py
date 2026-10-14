"""Dataset loader for PANORAMA ROI classification."""


def make_3d_dataset(dataset_str, data_items_path, npz_dir, fold=0, transform=None):
    try:
        dataset_name, split = dataset_str.split(":", 1)
    except ValueError as exc:
        raise ValueError("dataset_str must use the form 'dataset:split'.") from exc

    dataset_name = dataset_name.lower()
    if dataset_name == "panorama":
        if __package__:
            from .datasets.panorama import DataSet
        else:
            from datasets.panorama import DataSet
    else:
        raise NotImplementedError(
            f"Only PANORAMA is included in this release, got {dataset_name!r}."
        )

    return DataSet(
        data_items_path=data_items_path,
        npz_dir=npz_dir,
        split=split,
        fold=fold,
        transformer=transform,
    )

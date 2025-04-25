import torch
import torch.nn.functional as F


def crop(name, dataset_items):
    min_len = min(elem[name].shape[1] for elem in dataset_items)

    cropped = []
    for elem in dataset_items:
        total_len = elem[name].shape[1]
        start = (total_len - min_len) // 2
        end = start + min_len
        cropped.append(elem[name][:, start:end])

    return cropped

def collate_fn_test(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}

    result_batch["data_object_1"] = torch.vstack(crop("data_object_1", dataset_items))

    result_batch["data_object_2"] = torch.vstack(
        [elem["data_object_2"] for elem in dataset_items]
    )
    result_batch["index"] = torch.tensor([elem["index"] for elem in dataset_items])
    result_batch["test_pairs"] = torch.tensor([elem["test_pairs"] for elem in dataset_items])

    return result_batch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    if "test_pairs" in dataset_items[0]:
        return collate_fn_test(dataset_items)

    result_batch = {}

    result_batch["data_object"] = torch.vstack(
        [elem["data_object"] for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    if "emo_labels" in dataset_items[0]:
        result_batch["emo_labels"] = torch.tensor([elem["emo_labels"] for elem in dataset_items])

    return result_batch

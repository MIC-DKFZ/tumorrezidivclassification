import os
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO


class GlioblastomaDataset(Dataset):
    def __init__(
        self,
        data_root_path,
        split,
        fold,
        transform=None,
        binary_task=None,
        selected_modalities=None,
        split_file_name="splits.json",
        allowed_labels=None,
        excluded_labels=None,
        label_mapping=None,
    ):
        super().__init__()
        self.img_dir = Path(data_root_path)
        self.transform = transform
        self.selected_modalities = selected_modalities
        self.allowed_labels = set(allowed_labels) if allowed_labels is not None else None
        self.excluded_labels = set(excluded_labels) if excluded_labels is not None else None
        self.label_mapping = label_mapping

        label_file = self.img_dir / "labels.json"
        split_file = self.img_dir / split_file_name

        with open(split_file) as f:
            # Get the full list of cases for the current fold
            all_case_identifiers = json.load(f)[fold]["train" if split == "train" else "val"]

        with open(label_file) as f:
            labels_dict = json.load(f)

        # --- NEW: LOGIC FOR BINARY SUB-DATASET CREATION ---
        if binary_task:
            print(f"\n--- Creating a balanced binary sub-dataset for task '{binary_task}' ---")
            class_a, class_b = [int(c) for c in binary_task.split('_vs_')]

            # 1. Filter cases by the two selected classes
            cases_a = [case for case in all_case_identifiers if labels_dict.get(case) == class_a]
            cases_b = [case for case in all_case_identifiers if labels_dict.get(case) == class_b]

            # 2. Balance the classes by subsampling the majority class
            if len(cases_a) > len(cases_b):
                majority_cases, minority_cases = cases_a, cases_b
            else:
                majority_cases, minority_cases = cases_b, cases_a

            print(f"Original counts: {len(majority_cases)} (majority) vs. {len(minority_cases)} (minority)")
            random.shuffle(majority_cases)
            balanced_majority_cases = majority_cases[:len(minority_cases)]

            self.case_identifiers = balanced_majority_cases + minority_cases
            random.shuffle(self.case_identifiers)  # Shuffle the final combined list

            # 3. Remap labels to 0 and 1 for binary classification
            self.labels = [0 if labels_dict[case] == min(class_a, class_b) else 1 for case in self.case_identifiers]
            print(f"Final balanced count: {len(self.case_identifiers)} samples")

        else:
            # --- Original 3-class logic ---
            self.case_identifiers = []
            self.labels = []
            for case_id in all_case_identifiers:
                if case_id not in labels_dict:
                    continue

                label = labels_dict[case_id]
                if self.allowed_labels is not None and label not in self.allowed_labels:
                    continue
                if self.excluded_labels is not None and label in self.excluded_labels:
                    continue
                if self.label_mapping is not None:
                    label = self.label_mapping.get(label, self.label_mapping.get(str(label), label))

                self.case_identifiers.append(case_id)
                self.labels.append(label)

        print(f"Initialized {split} dataset for fold {fold} with {len(self.labels)} samples.")

    def __len__(self):
        return len(self.case_identifiers)

    def __getitem__(self, idx):
        case_id_str = self.case_identifiers[idx]
        modality_files = sorted(self.img_dir.glob(f"{case_id_str}_*.b2nd"))
        if not modality_files:
            raise FileNotFoundError(f"CRITICAL ERROR: No .b2nd files found for case '{case_id_str}'...")
        if self.selected_modalities is not None:
            try:
                modality_files = [modality_files[i] for i in self.selected_modalities]
            except IndexError as e:
                raise IndexError(
                    f"Case '{case_id_str}' has {len(modality_files)} modalities, "
                    f"but selected_modalities={self.selected_modalities} was requested."
                ) from e
        all_modalities = [Blosc2IO.load(p, mode="r")[0] for p in modality_files]
        img = np.stack(all_modalities, axis=0).squeeze(axis=1)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(**{"image": torch.from_numpy(img).float()})["image"]
        else:
            img = torch.from_numpy(img).float()
        return img, torch.tensor(label, dtype=torch.long)


class GlioblastomaDataModule(BaseDataModule):
    def __init__(self, **params):
        super(GlioblastomaDataModule, self).__init__(**params)
        self.name = params.get("name")
        self.data_root_dir = params.get("data_root_dir")
        self.fold = params.get("fold", 0)

        # --- NEW: Control flags from YAML ---
        self.weighted_sampling = params.get("weighted_sampling", True)
        self.binary_task = params.get("binary_task", None)
        self.selected_modalities = params.get("selected_modalities", None)
        self.split_file_name = params.get("split_file_name", "splits.json")
        self.allowed_labels = params.get("allowed_labels", None)
        self.excluded_labels = params.get("excluded_labels", None)
        self.label_mapping = params.get("label_mapping", None)

    def setup(self, stage: str):
        full_data_path = os.path.join(self.data_root_dir, self.name)

        # Create datasets, now passing the binary_task flag
        self.train_dataset = GlioblastomaDataset(
            data_root_path=full_data_path, split="train", transform=self.train_transforms, fold=self.fold,
            binary_task=self.binary_task, selected_modalities=self.selected_modalities,
            split_file_name=self.split_file_name, allowed_labels=self.allowed_labels,
            excluded_labels=self.excluded_labels, label_mapping=self.label_mapping
        )
        self.val_dataset = GlioblastomaDataset(
            data_root_path=full_data_path, split="val", transform=self.test_transforms, fold=self.fold,
            binary_task=self.binary_task, selected_modalities=self.selected_modalities,
            split_file_name=self.split_file_name, allowed_labels=self.allowed_labels,
            excluded_labels=self.excluded_labels, label_mapping=self.label_mapping
        )

        # --- NEW: Conditional Weighted Sampling ---
        # Only apply weighted sampling if enabled AND if it's not a (pre-balanced) binary task
        if self.weighted_sampling and self.binary_task is None:
            print("\nSetting up weighted random sampler for the training set...")
            train_labels = self.train_dataset.labels
            class_counts = np.bincount(train_labels)
            print(f"Training set class counts for this fold: {class_counts}")
            class_weights = 1. / (class_counts + 1e-6)
            print(f"Calculated class weights: {class_weights}")
            sample_weights = [class_weights[label] for label in train_labels]
            self.train_sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True
            )
            print("WeightedRandomSampler configured successfully.\n")
        else:
            print("\nWeighted random sampler is DISABLED for this run.")

import json
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


class Dataset:
    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)
        self.df: pd.DataFrame | None = None

    def get_df(self, unified: bool = True) -> pd.DataFrame:
        """Load the dataset and optionally return its unified representation."""
        self._load_dataset()
        dataframe = self._get_dataframe()
        return self._unify() if unified else dataframe

    def _get_dataframe(self) -> pd.DataFrame:
        """Return the loaded dataframe or report an invalid dataset state."""
        # Loaders initialize the dataframe lazily so constructing a dataset stays cheap.
        if self.df is None:
            raise RuntimeError("Dataset loading did not initialize the dataframe.")
        return self.df

    def get_train_test(
        self, split: float = 0.7, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return train and test dataframes for the dataset."""
        raise NotImplementedError()

    def _load_dataset(self) -> None:
        """Load raw annotations and metadata into ``self.df``."""
        raise NotImplementedError()

    def _unify(self) -> pd.DataFrame:
        """Convert the raw dataframe to the common dataset schema."""
        raise NotImplementedError()


class SPRSoundDataset(Dataset):
    def _load_dataset(self) -> None:
        if self.df is None:
            wav_paths = {f.stem: f for f in list(self.dataset_path.rglob("*.wav"))}
            annotations = []

            for json_file in self.dataset_path.rglob("*.json"):
                filename = json_file.stem

                with open(json_file, encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        events = data["event_annotation"]

                        for event in events:
                            event_type = event["type"].lower()

                            annotations.append(
                                {
                                    "start": float(event["start"]) / 1000,
                                    "end": float(event["end"]) / 1000,
                                    "event_type": event_type,
                                    "filename": filename,
                                    "wav_path": str(wav_paths[filename]),
                                    "patient_id": filename.split("_")[0],
                                    "patient_age": float(filename.split("_")[1]),
                                    "patient_gender": {"0": "Male", "1": "Female"}[
                                        filename.split("_")[2]
                                    ],
                                    "chest_location": filename.split("_")[3],
                                }
                            )
                    except Exception as e:
                        print(f"Błąd parsowania pliku {json_file.name}: {e}")
            df = pd.DataFrame(annotations)
            df_encoded = pd.get_dummies(df, columns=["event_type"], prefix="", prefix_sep="")
            self.df = df_encoded

    def _unify(self) -> pd.DataFrame:
        dataframe = self._get_dataframe()
        # Keep both wheeze and crackle labels available for the common schema.
        dataframe["crackles"] = dataframe[["fine crackle", "coarse crackle", "wheeze+crackle"]].any(
            axis=1
        )
        dataframe["wheezes"] = dataframe[["wheeze", "stridor", "wheeze+crackle"]].any(axis=1)

        columns = [
            "start",
            "end",
            "filename",
            "wav_path",
            "patient_id",
            "patient_age",
            "patient_gender",
            "chest_location",
            "wheezes",
            "crackles",
            "rhonchi",
        ]

        return cast(pd.DataFrame, dataframe[columns])

    def get_train_test(
        self, split: float = 0.7, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.get_df()

        gss = GroupShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        return train_df, test_df


class ICBHIDataset(Dataset):
    def __init__(self, dataset_path: str | Path, diagnosis_path: str | Path) -> None:
        super().__init__(dataset_path)
        self.diagnosis_path = Path(diagnosis_path)

    def _load_dataset(self) -> None:
        if self.df is None:
            annotations: list[pd.DataFrame] = []
            file_metadata: list[dict[str, Any]] = []

            for file in self.dataset_path.iterdir():
                parts = file.stem.split("_")
                if len(parts) != 5:
                    continue

                if file.suffix == ".wav":
                    file_metadata.append(
                        {
                            "patient_id": int(parts[0]),
                            "recording_id": parts[1],
                            "chest_location": parts[2],
                            "acquisition_mode": parts[3],
                            "recording_equipment": parts[4],
                            "filename": file.stem,
                            "wav_path": str(file),
                        }
                    )
                elif file.suffix == ".txt":
                    try:
                        anno_df = pd.read_csv(
                            file, sep="\t", names=["start", "end", "crackles", "wheezes"]
                        )
                        anno_df["filename"] = file.stem
                        if anno_df is not None:
                            annotations.append(anno_df)
                    except Exception as e:
                        print(f"Pominięto uszkodzony plik adnotacji {file.name}: {e}")

            df = pd.concat(annotations, ignore_index=True)
            metadata_df = pd.DataFrame(file_metadata)
            patients_df = pd.read_csv(
                self.diagnosis_path, sep="\t", names=["patient_id", "diagnosis"]
            )
            merged_metadata = pd.merge(metadata_df, patients_df, on="patient_id", how="left")
            df = pd.merge(df, merged_metadata, on="filename", how="left")
            self.df = df

    def _unify(self) -> pd.DataFrame:
        return self._get_dataframe()

    def get_train_test(
        self, split: float = 0.7, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.get_df()

        gss = GroupShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        return train_df, test_df


class HFLungSoundDataset(Dataset):
    def __init__(self, dataset_path: str | Path) -> None:
        super().__init__(dataset_path)

        self.dataset1_path = self.dataset_path / "HF_Lung_V1-master"
        self.dataset2_path = self.dataset_path / "HF_Lung_V1_IP-main"

        self.train_path1 = self.dataset1_path / "train"
        self.test_path1 = self.dataset1_path / "test"
        self.train_path2 = self.dataset2_path / "train"
        self.test_path2 = self.dataset2_path / "test"

    def _load_dataset(self) -> None:
        if self.df is not None:
            return

        annotations: list[dict[str, Any]] = []
        file_metadata: list[dict[str, Any]] = []

        paths_to_search = [self.train_path1, self.test_path1, self.train_path2, self.test_path2]
        all_paths = chain.from_iterable(p.rglob("*") for p in paths_to_search if p.exists())

        for file in all_paths:
            if not file.is_file():
                continue

            filename_stem = file.stem
            parts = filename_stem.split("_")

            split_val = "train" if "train" in file.parts else "test"

            if file.suffix == ".wav":
                if len(parts) == 5:
                    audio_type, date_str, h, m, s = parts
                    try:
                        date_obj = datetime(
                            int(date_str[:4]),
                            int(date_str[4:6]),
                            int(date_str[6:8]),
                            int(h),
                            int(m),
                            int(s),
                        )
                        file_metadata.append(
                            {
                                "recording_equipment": audio_type,
                                "date": date_obj,
                                "chest_location": None,
                                "truncation": 1,
                                "filename": filename_stem,
                                "wav_path": str(file),
                                "split": split_val,
                            }
                        )
                    except ValueError as e:
                        print(f"Błąd parsowania daty w pliku {file.name}: {e}")

                elif len(parts) == 3:
                    audio_type, date_loc_str, truncation = parts
                    date_loc_parts = date_loc_str.split("-")
                    try:
                        args = [int(x) for x in date_loc_parts[:-1]]
                        date_obj = datetime(
                            year=args[0],
                            month=args[1],
                            day=args[2],
                            hour=args[3] if len(args) > 3 else 0,
                            minute=args[4] if len(args) > 4 else 0,
                            second=args[5] if len(args) > 5 else 0,
                        )
                        file_metadata.append(
                            {
                                "recording_equipment": audio_type,
                                "date": date_obj,
                                "chest_location": date_loc_parts[-1],
                                "truncation": int(truncation),
                                "filename": filename_stem,
                                "wav_path": str(file),
                                "split": split_val,
                            }
                        )
                    except ValueError as e:
                        print(f"Błąd parsowania daty w pliku {file.name}: {e}")

            elif file.suffix == ".txt":
                if len(parts) not in (4, 6):
                    continue
                try:
                    phases: list[dict[str, Any]] = []
                    annotations_raw: list[dict[str, Any]] = []

                    with open(file, encoding="utf-8") as f:
                        for line in f:
                            line_parts = line.strip().split()
                            if len(line_parts) < 3:
                                continue
                            t_type, start_str, end_str = line_parts[0], line_parts[1], line_parts[2]

                            try:
                                start_sec = float(start_str)
                            except ValueError:
                                start_sec = pd.to_timedelta(start_str).total_seconds()

                            try:
                                end_sec = float(end_str)
                            except ValueError:
                                end_sec = pd.to_timedelta(end_str).total_seconds()

                            item = {"type": t_type, "start": start_sec, "end": end_sec}
                            if t_type in ("I", "E"):
                                phases.append(item)
                            else:
                                annotations_raw.append(item)

                    if not phases:
                        continue

                    phases.sort(key=lambda x: x["start"])

                    cycles = []
                    current_start, current_end = None, None

                    for phase in phases:
                        if phase["type"] == "I":
                            if current_start is not None:
                                cycles.append({"start": current_start, "end": current_end})
                            current_start = phase["start"]
                            current_end = phase["end"]
                        elif phase["type"] == "E":
                            if current_start is None:
                                current_start = phase["start"]
                            current_end = (
                                max(current_end, phase["end"]) if current_end else phase["end"]
                            )

                    if current_start is not None:
                        cycles.append({"start": current_start, "end": current_end})

                    if not cycles:
                        continue

                    file_name_val = (
                        filename_stem.rsplit("_", 1)[0] if len(parts) in (4, 6) else filename_stem
                    )

                    type_map = {
                        "d": "crackles",
                        "wheeze": "wheezes",
                        "rhonchi": "rhonchi",
                        "stridor": "stridor",
                    }

                    file_cycles = []
                    for cycle in cycles:
                        c_dict = {
                            "start": cycle["start"],
                            "end": cycle["end"],
                            "filename": file_name_val,
                            "crackles": 0,
                            "wheezes": 0,
                            "rhonchi": 0,
                            "stridor": 0,
                        }

                        for anomaly in annotations_raw:
                            target_col = type_map.get(str(anomaly["type"]).lower())
                            if target_col and (
                                cycle["start"] < anomaly["end"] and cycle["end"] > anomaly["start"]
                            ):
                                c_dict[target_col] = 1

                        file_cycles.append(c_dict)

                    annotations.extend(file_cycles)

                except Exception as e:
                    print(f"Pominięto uszkodzony plik adnotacji {file.name}: {e}")

        df_all = pd.DataFrame(annotations)
        metadata_df = pd.DataFrame(file_metadata)
        if not metadata_df.empty and "truncation" in metadata_df.columns:
            metadata_df["truncation"] = metadata_df["truncation"].astype(int)

        if not df_all.empty and not metadata_df.empty:
            self.df = pd.merge(df_all, metadata_df, on="filename", how="left")
        else:
            self.df = pd.DataFrame()

    def _unify(self) -> pd.DataFrame:
        dataframe = self._get_dataframe()
        dataframe["crackles"] = dataframe[["crackles"]].any(axis=1)
        dataframe["wheezes"] = dataframe[["wheezes", "stridor"]].any(axis=1)
        columns = [
            "start",
            "end",
            "filename",
            "wav_path",
            "recording_equipment",
            "date",
            "truncation",
            "chest_location",
            "split",
            "wheezes",
            "crackles",
            "rhonchi",
        ]

        return cast(pd.DataFrame, dataframe[columns])

    def get_train_test(
        self, split: float = 0.7, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.get_df()

        train_df = cast(
            pd.DataFrame, df[df["split"] == "train"].drop(columns=["split"], errors="ignore")
        )
        test_df = cast(
            pd.DataFrame, df[df["split"] == "test"].drop(columns=["split"], errors="ignore")
        )

        return train_df, test_df

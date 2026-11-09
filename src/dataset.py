from pathlib import Path

import librosa

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
from itertools import chain
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit

class Dataset:
    def __init__(self, dataset_path, cache_path=None):
        self.dataset_path = Path(dataset_path)
        if cache_path is not None:
            self.cache_path = Path(cache_path)
        else:
            self.cache_path = cache_path
        self.df = None

    def _get_cache(self):
            table = pq.read_table(self.cache_path)
            df = table.to_pandas()
            self.df = df
    
    def _cache_df(self):
        table = pa.Table.from_pandas(self.df) 
        pq.write_table(table, self.cache_path)

    def get_df(self, unified=True):
        if self.cache_path is None:
            self._load_dataset()
        else:
            if self.cache_path.exists():
                self._get_cache()
            else:
                self._load_dataset()
                self._cache_df()
        if unified:
            return self._unify()
        return self.df

    def get_train_test(self):
        raise NotImplementedError()

    def _load_dataset(self):
        raise NotImplementedError()

    def _unify(self):
        raise NotImplementedError()

    def load_audio(self, record, sr=None):
        if isinstance(record, (int, np.integer)):
            df = self.get_df(unified=True)
            record = df.iloc[record]
        
        wav_path = record['wav_path'] if isinstance(record, (pd.Series, dict)) else self.get_df(unified=True).iloc[record]['wav_path']
        return librosa.load(wav_path, sr=sr)

    def load_audio_fragment(self, record, start=None, end=None, sr=None):
        if isinstance(record, (int, np.integer)):
            df = self.get_df(unified=True)
            record = df.iloc[record]
            
        if isinstance(record, (pd.Series, dict)):
            wav_path = record['wav_path']
            if start is None:
                start = record.get('start', 0.0)
            if end is None:
                end = record.get('end', None)
        else:
            df = self.get_df(unified=True)
            row = df.iloc[record]
            wav_path = row['wav_path']
            start = start if start is not None else row.get('start', 0.0)
            end = end if end is not None else row.get('end', None)

        offset = float(start) if start is not None else 0.0
        duration = float(end - start) if (end is not None and start is not None) else None

        return librosa.load(wav_path, sr=sr, offset=offset, duration=duration)

class SPRSoundDataset(Dataset):

    def _load_dataset(self):
        if self.df is None:
            wav_paths = { f.stem : f for f in list(self.dataset_path.rglob("*.wav"))}
            annotations = []
                
            for json_file in self.dataset_path.rglob("*.json"):
                filename = json_file.stem
                    
                with open(json_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)

                        events = data['event_annotation'] 
                        
                        for event in events:
                            event_type = event['type'].lower()
                            
                            annotations.append({
                                "start": float(event['start']) / 1000,
                                "end": float(event['end']) / 1000,
                                "event_type": event_type,
                                "filename": filename,
                                "wav_path": str(wav_paths[filename]),
                                "patient_id": filename.split('_')[0],
                                'patient_age': float(filename.split('_')[1]),
                                'patient_gender': {'0': 'Male', '1': 'Female'}[filename.split('_')[2]],
                                'chest_location': filename.split('_')[3]
                            })
                    except Exception as e:
                        print(f"Błąd parsowania pliku {json_file.name}: {e}")
            df = pd.DataFrame(annotations)
            df_encoded = pd.get_dummies(df, columns=['event_type'], prefix='', prefix_sep='')
            self.df = df_encoded

    def _unify(self):
        self.df['crackles'] = self.df[['fine crackle', 'coarse crackle', 'wheeze+crackle']].any(axis=1)
        self.df['wheezes'] = self.df[['wheeze', 'rhonchi', 'stridor', 'wheeze+crackle']].any(axis=1)
        self.df = self.df[self.df['chest_location'].isin(['p1', 'p2', 'p3', 'p4'])]

        columns = ['start','end','filename','wav_path','patient_id','patient_age','patient_gender','chest_location','wheezes','crackles']
        return self.df[columns]

    def get_train_test(self, split=0.7, random_state=42):
        df = self.get_df(unified=True)
            
        gss = GroupShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, groups=df['patient_id']))
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        return train_df, test_df


class ICBHIDataset(Dataset):
    def __init__(self, dataset_path, diagnosis_path, cache_path=None):
        super().__init__(dataset_path, cache_path)
        self.diagnosis_path = diagnosis_path

    def _load_dataset(self):
        if self.df is None:
            annotations = []
            file_metadata = []
        
            for file in self.dataset_path.iterdir():
                parts = file.stem.split("_")
                if len(parts) != 5:
                    continue
        
                if file.suffix == ".wav":
                    file_metadata.append({"patient_id": int(parts[0]),
                                        "recording_id": parts[1],
                                        "chest_location": parts[2],
                                        "acquisition_mode": parts[3],
                                        "recording_equipment": parts[4],
                                        "filename": file.stem,
                                        "wav_path": str(file)})
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
            merged_metadata = pd.merge(
                metadata_df, patients_df, on="patient_id", how="left"
            )
            df = pd.merge(df, merged_metadata, on="filename", how="left")
            self.df = df
    
    def _unify(self):
        return self.df

    def get_train_test(self, split=0.7, random_state=42):
        df = self.get_df(unified=True)
            
        gss = GroupShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, groups=df['patient_id']))
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        return train_df, test_df

class HFLungSoundDataset(Dataset):
    def __init__(self, dataset_path, cache_path=None):
        super().__init__(dataset_path, cache_path)

        self.dataset1_path = self.dataset_path / "HF_Lung_V1-master"
        self.dataset2_path = self.dataset_path / "HF_Lung_V1_IP-main"

        self.train_path1 = self.dataset1_path / "train"
        self.test_path1 = self.dataset1_path / "test"
        self.train_path2 = self.dataset2_path / "train"
        self.test_path2 = self.dataset2_path / "test"

    def _load_dataset(self):
        if self.df is not None:
            return

        annotations = []
        file_metadata = []

        paths_to_search = [self.train_path1, self.test_path1, self.train_path2, self.test_path2]
        all_paths = chain.from_iterable(p.rglob("*") for p in paths_to_search if p.exists())

        for file in all_paths:
            if not file.is_file():
                continue

            filename_stem = file.stem
            parts = filename_stem.split("_")
            
            # Określenie, czy plik należy do train, czy test na podstawie struktury ścieżki
            split_val = "train" if "train" in file.parts else "test"

            if file.suffix == ".wav":
                if len(parts) == 5:
                    audio_type, date_str, h, m, s = parts
                    try:
                        date_obj = datetime(
                            int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]),
                            int(h), int(m), int(s)
                        )
                        file_metadata.append({
                            "recording_equipment": audio_type,
                            "date": date_obj,
                            "chest_location": None,
                            "truncation": 1,
                            "filename": filename_stem, 
                            "wav_path": str(file),
                            "split": split_val
                        })
                    except ValueError as e:
                        print(f"Błąd parsowania daty w pliku {file.name}: {e}")
                        
                elif len(parts) == 3:
                    audio_type, date_loc_str, truncation = parts
                    date_loc_parts = date_loc_str.split("-")
                    try:
                        args = [int(x) for x in date_loc_parts[:-1]]
                        date_obj = datetime(*args)
                        file_metadata.append({
                            "recording_equipment": audio_type,
                            "date": date_obj,
                            "chest_location": date_loc_parts[-1],
                            "truncation": truncation,
                            "filename": filename_stem,
                            "wav_path": str(file),
                            "split": split_val
                        })
                    except ValueError as e:
                        print(f"Błąd parsowania daty w pliku {file.name}: {e}")

            elif file.suffix == ".txt":
                if len(parts) not in (4, 6):
                    continue
                try:
                    phases = []
                    annotations_raw = []
                    
                    with open(file, 'r', encoding='utf-8') as f:
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
                            current_end = max(current_end, phase["end"]) if current_end else phase["end"]

                    if current_start is not None:
                        cycles.append({"start": current_start, "end": current_end})

                    if not cycles:
                        continue

                    file_name_val = filename_stem.rsplit('_', 1)[0] if len(parts) in (4, 6) else filename_stem
                    
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
                            "stridor": 0
                        }

                        for anomaly in annotations_raw:
                            target_col = type_map.get(str(anomaly["type"]).lower())
                            if target_col:
                                if cycle["start"] < anomaly["end"] and cycle["end"] > anomaly["start"]:
                                    c_dict[target_col] = 1

                        file_cycles.append(c_dict)

                    annotations.extend(file_cycles)

                except Exception as e:
                    print(f"Pominięto uszkodzony plik adnotacji {file.name}: {e}")

        df_all = pd.DataFrame(annotations)
        metadata_df = pd.DataFrame(file_metadata)

        if not df_all.empty and not metadata_df.empty:
            self.df = pd.merge(df_all, metadata_df, on="filename", how="left")
        else:
            self.df = pd.DataFrame()

    def _unify(self):
        self.df['crackles'] = self.df[['crackles']].any(axis=1)
        self.df['wheezes'] = self.df[['wheezes', 'rhonchi', 'stridor']].any(axis=1)
        columns = [
            'start', 'end', 'filename', 'wav_path', 
            'recording_equipment', 'date', 'truncation', 
            'chest_location', 'split', 'wheezes', 'crackles'
        ]
        return self.df[columns]

    def get_train_test(self):
        train_df = self.df[self.df['split'] == 'train'].drop(columns=['split'])
        test_df = self.df[self.df['split'] == 'test'].drop(columns=['split'])
        
        return train_df, test_df
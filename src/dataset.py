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
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.df = None

    def _get_cache(self):
        table = pq.read_table(self.cache_path)
        self.df = table.to_pandas()
    
    def _cache_df(self):
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(self.df) 
            pq.write_table(table, self.cache_path)

    def _get_features(self):
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        rms_dict = {}
        zcr_dict = {}
        spectral_centroid_dict = {}
        spectral_flatness_dict = {}
        spectral_rolloff_dict = {}
        spectral_bandwidth_dict = {}
        crest_factor_dict = {}

        for wav_path in self.df['wav_path'].unique():
            y, sr = librosa.load(wav_path, sr=None)
            
            cycles_for_file = self.df[self.df['wav_path'] == wav_path]
            
            for cycle in cycles_for_file.itertuples():
                idx = cycle.Index
                
                start_sample = int(cycle.start * sr)
                end_sample = int(cycle.end * sr)
                
                y_cycle = y[start_sample:end_sample]
                
                if len(y_cycle) == 0:
                    continue

                rms = np.mean(librosa.feature.rms(y=y_cycle))
                rms_dict[idx] = rms
                zcr_dict[idx] = np.mean(librosa.feature.zero_crossing_rate(y=y_cycle))
                crest_factor_dict[idx] = np.max(np.abs(y_cycle)) / (rms + 1e-8)

                spectral_centroid_dict[idx] = np.mean(librosa.feature.spectral_centroid(y=y_cycle, sr=sr))
                spectral_bandwidth_dict[idx] = np.mean(librosa.feature.spectral_bandwidth(y=y_cycle, sr=sr))
                spectral_rolloff_dict[idx] = np.mean(librosa.feature.spectral_rolloff(y=y_cycle, sr=sr, roll_percent=0.85))
                spectral_flatness_dict[idx] = np.mean(librosa.feature.spectral_flatness(y=y_cycle))

        self.df = self.df.copy()
        self.df['audio_rms'] = self.df.index.map(rms_dict)
        self.df['audio_zcr'] = self.df.index.map(zcr_dict)
        self.df['crest_factor'] = self.df.index.map(crest_factor_dict)
        self.df['spectral_centroid'] = self.df.index.map(spectral_centroid_dict)
        self.df['spectral_bandwidth'] = self.df.index.map(spectral_bandwidth_dict)
        self.df['spectral_rolloff'] = self.df.index.map(spectral_rolloff_dict)
        self.df['spectral_flatness'] = self.df.index.map(spectral_flatness_dict)

    def get_df(self, unified=True, extract_features=False):
        if self.cache_path is not None and self.cache_path.exists():
            self._get_cache()
        else:
            self._load_dataset()
            self._cache_df()

        feature_cols = [
            'audio_rms', 'audio_zcr', 'crest_factor', 
            'spectral_centroid', 'spectral_bandwidth', 
            'spectral_rolloff', 'spectral_flatness'
        ]
        
        has_features = all(col in self.df.columns for col in feature_cols)

        if extract_features and not has_features:
            self._get_features()
            self._cache_df()

        if unified:
            return self._unify(extract_features=extract_features)
        
        if not extract_features:
            return self.df.drop(columns=[col for col in feature_cols if col in self.df.columns], errors='ignore')
        return self.df

    def get_train_test(self):
        raise NotImplementedError()

    def _load_dataset(self):
        raise NotImplementedError()

    def _unify(self, extract_features=False):
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

    def _unify(self, extract_features=False):
        self.df['crackles'] = self.df[['fine crackle', 'coarse crackle', 'wheeze+crackle']].any(axis=1)
        self.df['wheezes'] = self.df[['wheeze', 'rhonchi', 'stridor', 'wheeze+crackle']].any(axis=1)
        self.df = self.df[self.df['chest_location'].isin(['p1', 'p2', 'p3', 'p4'])]

        columns = ['start','end','filename','wav_path','patient_id','patient_age','patient_gender','chest_location','wheezes','crackles']
        feature_cols = ['audio_rms', 'audio_zcr', 'crest_factor', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness']
        
        if extract_features:
            existing_features = [col for col in feature_cols if col in self.df.columns]
            return self.df[columns + existing_features]
        return self.df[columns]

    def get_train_test(self, split=0.7, random_state=42, extract_features=False):
        df = self.get_df(unified=True, extract_features=extract_features)
            
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
    
    def _unify(self, extract_features=False):
        feature_cols = ['audio_rms', 'audio_zcr', 'crest_factor', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness']
        if not extract_features:
            return self.df.drop(columns=[col for col in feature_cols if col in self.df.columns], errors='ignore')
        return self.df

    def get_train_test(self, split=0.7, random_state=42, extract_features=False):
        df = self.get_df(unified=True, extract_features=extract_features)
            
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

    def _unify(self, extract_features=False):
        self.df['crackles'] = self.df[['crackles']].any(axis=1)
        self.df['wheezes'] = self.df[['wheezes', 'rhonchi', 'stridor']].any(axis=1)
        columns = [
            'start', 'end', 'filename', 'wav_path', 
            'recording_equipment', 'date', 'truncation', 
            'chest_location', 'split', 'wheezes', 'crackles'
        ]
        feature_cols = ['audio_rms', 'audio_zcr', 'crest_factor', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness']
        
        if extract_features:
            existing_features = [col for col in feature_cols if col in self.df.columns]
            return self.df[columns + existing_features]
        return self.df[columns]

    def get_train_test(self, extract_features=False):
        df = self.get_df(unified=True, extract_features=extract_features)
        
        train_df = df[df['split'] == 'train'].drop(columns=['split'], errors='ignore')
        test_df = df[df['split'] == 'test'].drop(columns=['split'], errors='ignore')
        
        return train_df, test_df


def transform_spr(df):
    df = df.copy()
    df['dataset_source'] = 'SPRSound'
    base_cols = ['start', 'end', 'filename', 'wav_path', 'chest_location', 'wheezes', 'crackles', 'dataset_source']
    feature_cols = [c for c in df.columns if c not in base_cols]
    return df[base_cols + feature_cols]

def transform_icbhi(df):
    df = df.copy()
    df['dataset_source'] = 'ICBHI'
    if 'chest_location' not in df.columns:
        df['chest_location'] = None
    return df

def transform_hf(df):
    df = df.copy()
    df['dataset_source'] = 'HFLung'
    base_cols = ['start', 'end', 'filename', 'wav_path', 'chest_location', 'wheezes', 'crackles', 'dataset_source']
    feature_cols = [c for c in df.columns if c not in base_cols]
    return df[base_cols + feature_cols]


def get_train_test(dataset1, dataset2, dataset3, split=0.7, random_state=42, extract_features=False):
    train_list = []
    test_list = []

    datasets = [dataset1, dataset2, dataset3]
    
    for dataset in datasets:
        if isinstance(dataset, HFLungSoundDataset):
            train, test = dataset.get_train_test(extract_features=extract_features)
        else:
            train, test = dataset.get_train_test(split=split, random_state=random_state, extract_features=extract_features)
        
        if isinstance(dataset, SPRSoundDataset):
            train_transformed = transform_spr(train)
            test_transformed = transform_spr(test)
        elif isinstance(dataset, ICBHIDataset):
            train_transformed = transform_icbhi(train)
            test_transformed = transform_icbhi(test)
        elif isinstance(dataset, HFLungSoundDataset):
            train_transformed = transform_hf(train)
            test_transformed = transform_hf(test)
        else:
            raise TypeError(f"Nieznany typ datasetu: {type(dataset)}")
            
        train_list.append(train_transformed)
        test_list.append(test_transformed)

    combined_train = pd.concat(train_list, ignore_index=True)
    combined_test = pd.concat(test_list, ignore_index=True)

    return combined_train, combined_test
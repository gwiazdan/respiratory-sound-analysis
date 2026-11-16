import librosa
import numpy as np
import pandas as pd

from src.audio import AudioPreprocessor
from src.dataset import Dataset, HFLungSoundDataset, ICBHIDataset, SPRSoundDataset


class ExperimentalDataset(Dataset):
    """To be deleted"""

    def _get_features(self):
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        features_list = []

        for wav_path in self.df["wav_path"].unique():
            old_y, sr = librosa.load(wav_path, sr=4000)
            preprocessor = AudioPreprocessor(sr=4000)

            cycles_for_file = self.df[self.df["wav_path"] == wav_path]
            y = preprocessor(old_y)
            for cycle in cycles_for_file.itertuples():
                idx = cycle.Index

                start_sample = int(cycle.start * sr)
                end_sample = int(cycle.end * sr)

                y_cycle = y[start_sample:end_sample]

                if len(y_cycle) < 128:
                    continue

                n_fft = min(2048, 2 ** int(np.floor(np.log2(len(y_cycle)))))
                if n_fft < 64:
                    n_fft = 64

                hop_length = max(1, n_fft // 4)

                def compute_stats(arr, prefix):
                    return {
                        f"{prefix}_mean": float(np.mean(arr)),
                        f"{prefix}_std": float(np.std(arr)),
                    }

                rms_arr = librosa.feature.rms(y=y_cycle, frame_length=n_fft, hop_length=hop_length)
                zcr_arr = librosa.feature.zero_crossing_rate(
                    y=y_cycle, frame_length=n_fft, hop_length=hop_length
                )

                rms_mean_val = np.mean(rms_arr)
                crest_factor = float(np.max(np.abs(y_cycle)) / (rms_mean_val + 1e-8))

                spec_cent = librosa.feature.spectral_centroid(
                    y=y_cycle, sr=sr, n_fft=n_fft, hop_length=hop_length
                )
                spec_band = librosa.feature.spectral_bandwidth(
                    y=y_cycle, sr=sr, n_fft=n_fft, hop_length=hop_length
                )
                spec_roll = librosa.feature.spectral_rolloff(
                    y=y_cycle, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
                )
                spec_flat = librosa.feature.spectral_flatness(
                    y=y_cycle, n_fft=n_fft, hop_length=hop_length
                )

                mfcc = librosa.feature.mfcc(
                    y=y_cycle, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length
                )

                row_feats = {"index": idx, "crest_factor": crest_factor}
                row_feats.update(compute_stats(rms_arr, "audio_rms"))
                row_feats.update(compute_stats(zcr_arr, "audio_zcr"))
                row_feats.update(compute_stats(spec_cent, "spectral_centroid"))
                row_feats.update(compute_stats(spec_band, "spectral_bandwidth"))
                row_feats.update(compute_stats(spec_roll, "spectral_rolloff"))
                row_feats.update(compute_stats(spec_flat, "spectral_flatness"))

                for i in range(mfcc.shape[0]):
                    row_feats[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
                    row_feats[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))

                features_list.append(row_feats)

        if features_list:
            feat_df = pd.DataFrame(features_list).set_index("index")
            self.df = self.df.copy()
            cols_to_drop = [c for c in feat_df.columns if c in self.df.columns]
            if cols_to_drop:
                self.df = self.df.drop(columns=cols_to_drop)
            self.df = self.df.join(feat_df, how="left")


HFLungSoundDataset.__bases__ = (ExperimentalDataset,)
SPRSoundDataset.__bases__ = (ExperimentalDataset,)
ICBHIDataset.__bases__ = (ExperimentalDataset,)

from typing import Any
import numpy as np
from tqdm import trange
from scipy.signal import find_peaks
from audiomentations import Compose, TimeStretch, PitchShift
from audiomentations import ClippingDistortion, Gain
from audiomentations import GainTransition, Reverse, Compose
from audiomentations import AddGaussianNoise, TimeStretch, PitchShift



class Augment:

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Pause, don't use
        gauss_noise = AddGaussianNoise(min_amplitude=0.1, max_amplitude=1.2, p=0.5)
        time_stretch = TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5)
        pitch_shift = PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
        compose = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        ])
        """ 
        clipping = ClippingDistortion(
            min_percentile_threshold=1,
            max_percentile_threshold=2,
            p=1.0
        )
        gain = Gain(
            min_gain_in_db=-2.0,
            max_gain_in_db=-1.1,
            p=1.0
        )
        gain_transition = GainTransition(
            min_gain_in_db=1.1,
            max_gain_in_db=2.0,
            p=1.0
        )
        reverse = Reverse(p=1.0)
        self.augments = [clipping, gain, gain_transition, reverse]
        self.features = features
        self.labels = labels

    @staticmethod
    def estimate_peaks(data: np.ndarray):
        peaks, _  = find_peaks(data)
        return peaks

    @staticmethod
    def check_errors(data, data_augment, peaks_fn=None):
        assert data.shape == data_augment.shape
        data_mean = data[:,:-1].mean(axis=-1)
        data_aug_mean = data_augment[:,:-1].mean(axis = -1)
        # Calculate number of peaks point in time series data
        data_peaks = peaks_fn(data_mean)
        data_aug_peaks = peaks_fn(data_aug_mean)
        if abs(len(data_peaks) - len(data_aug_peaks))  <= 1:
            return True
        else:
            return False

    def augment_signal(
        self,
        features,
        labels,
        augments
    ):
        features_augment = np.array([])
        labels_augment = np.array([])
        errors_data = []
        for i in trange(len(features)):
            X_vsl = features[i, :, 0]
            Y_vsl = features[i, :, 1]
            Z_vsl = features[i, :, 2]
            label_pos = features[i, :, 3]
            aug_errors = []
            for augment_method in augments:
                X_aug = augment_method(
                    samples=X_vsl,
                    sample_rate=8000
                )
                Y_aug = augment_method(
                    samples=Y_vsl,
                    sample_rate=8000
                )
                Z_aug = augment_method(
                    samples=Z_vsl,
                    sample_rate=8000
                )
                aug_data = np.transpose(np.array([X_aug, Y_aug, Z_aug, label_pos]))
                error = self.check_errors(
                    data=features[i],
                    data_augment=aug_data,
                    peaks_fn=self.estimate_peaks
                )
                aug_errors.append(error)
                if features_augment.shape[0] == 0:
                    features_augment = np.expand_dims(np.transpose(np.array([X_aug, Y_aug, Z_aug, label_pos])), axis=0)
                    labels_augment = np.expand_dims(labels[i], axis=0)
                else:
                    features_augment = np.concatenate([features_augment,np.expand_dims(np.transpose(np.array([X_aug, Y_aug, Z_aug, label_pos])), axis=0)], axis = 0)
                    labels_augment = np.concatenate([labels_augment, np.expand_dims(labels[i],axis=0)], axis=0)
            errors_data.append(aug_errors)
        return features_augment, labels_augment, errors_data

    def __call__(self, augment_times):
        features_augment = self.features.copy() 
        labels_augment = self.labels.copy()
        print(f"Prepare augment {augment_times} times...")
        for _ in range(augment_times):
            features_augment, labels_augment, errors_data = self.augment_signal(
                features_augment, labels_augment, self.augments
            )
        errors_augment = np.array(errors_data, dtype=np.float32).mean(axis=0)
        augment_names = []
        for name in self.augments:
            augment_names.append(name.__class__.__name__)
        print(f"\nError augment methods:")
        for error, augment_name in zip(errors_augment, augment_names):
            print("- {}: {}".format(augment_name, error))    
        return features_augment, labels_augment

        
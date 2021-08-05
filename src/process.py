import os
from typing import List, Union, Callable, Any, Optional, Tuple, Dict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

import pydicom
import tensorflow as tf
import tensorflow_io as tfio

from src.types import Patient, Image, Attribute
from src.otsu import remove_noise


def read_metadata(path: str) -> Dict[Attribute, str]:
    metadata = pydicom.read_file(path)
    dic: Dict[Attribute, str] = {}
    for attr in Attribute:
        try:
            dic[attr] = getattr(metadata, attr.value)
        except AttributeError:
            pass
    return dic


def decode_patient(fold_patient: str, label: str, path: str, args,
                   kwargs) -> Patient:
    new_patient = Patient(id=fold_patient + label[0], label=label)
    patient_path = os.path.join(path, label, 'DICOM', fold_patient)
    images_path = [
        os.path.join(patient_path, img_path)
        for img_path in os.listdir(patient_path)
    ]
    images_file = [
        DicomDataset._read_file(img_path) for img_path in images_path
    ]
    images = [
        Image(img_path, DicomDataset._decode_dicom(img_file, *args, **kwargs))
        for img_file, img_path in zip(images_file, images_path)
    ]
    setattr(new_patient, 'images', images)
    if len(images) > 0:
        metadata = read_metadata(images[0].path)
        setattr(new_patient, 'metadata', metadata)
    return new_patient


def func_wrapper(funct, img: Image, only_img: bool, kwargs):
    if only_img:
        if len(img.image.shape) in [3, 4]:
            return funct(img.image, **kwargs)
        else:
            return img
    else:
        return funct(img, **kwargs)


class DicomDataset:
    def __init__(self, path: str, multi: bool = False, *args, **kwargs):
        if multi:
            self.patients = self.decode_dicom_multi(path, *args, **kwargs)
        else:
            self.patients = self.decode_dicoms(path, *args, **kwargs)

    @staticmethod
    def _decode_dicom(file_tf: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return tfio.image.decode_dicom_image(file_tf, *args, **kwargs)

    @staticmethod
    def _read_file(file_name: str) -> tf.Tensor:
        return tf.io.read_file(file_name)

    def decode_dicoms(self, path: str, *args, **kwargs) -> List[Patient]:
        patients = []
        for label in ['FRACTURE', 'NO_FRACTURE']:
            for fold_patient in tqdm(os.listdir(
                    os.path.join(path, label, 'DICOM')),
                                     desc='Patients ' + label):
                new_patient = decode_patient(fold_patient, label, path, args,
                                             kwargs)
                patients.append(new_patient)
        return patients

    def decode_dicom_multi(self, path: str, *args, **kwargs) -> List[Patient]:
        patients: List[Patient] = []
        for label in ['FRACTURE', 'NO_FRACTURE']:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                listdir = os.listdir(os.path.join(path, label, 'DICOM'))
                for patient in tqdm(pool.imap(
                        partial(decode_patient,
                                label=label,
                                path=path,
                                args=args,
                                kwargs=kwargs), listdir),
                                    desc='Patients ' + label,
                                    total=len(listdir)):
                    patients.append(patient)
        return patients

    def resize(self,
               new_height: int,
               new_width: int,
               method: Union[tf.image.ResizeMethod,
                             str] = tf.image.ResizeMethod.BILINEAR,
               preserve_aspect_ratio: bool = False):
        params = {
            "size": (new_height, new_width),
            "method": method,
            "preserve_aspect_ratio": preserve_aspect_ratio,
        }
        self.apply(tf.image.resize, params)

    def change_contrast(self, contrast_factor: float):
        params = {"contrast_factor": contrast_factor}
        self.apply(tf.image.adjust_contrast, params)

    def remove_noise(self, size_ratio: Optional[float] = None):
        images: List[Image] = []
        indexes: List[Tuple[int, int]] = []
        for i, patient in enumerate(self.patients):
            if len(patient.images) == 0:
                continue
            for j, image in enumerate(patient.images):
                images.append(image)
                indexes.append((i, j))

        with mp.Pool(processes=mp.cpu_count()) as pool:
            for idx, res in tqdm(enumerate(
                    pool.imap(partial(remove_noise, size_ratio=size_ratio),
                              images)),
                                 total=len(images)):
                i, j = indexes[idx]
                self.patients[i].images[j] = res

    def invert_color(self):
        for patient in self.patients:
            for image in patient.images:
                image.invert_color()

    def apply(self, func: Callable[[tf.Tensor, Any], tf.Tensor], kwargs):
        for i in range(len(self.patients)):
            for j in range(len(self.patients[i].images)):
                img = self.patients[i].images[j].image
                if len(img.numpy().shape) in [3, 4]:
                    self.patients[i].images[j].image = func(img, **kwargs)

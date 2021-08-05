from dataclasses import dataclass, field, fields
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict
from functools import reduce

import tensorflow as tf

THRESHOLD_RATIO = 0.35


class Pos(Enum):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOT_LEFT = 2
    BOT_RIGHT = 3


class Color(Enum):
    BLACK = 0
    WHITE = 1


@dataclass
class Edge:
    val: tf.Tensor
    position: Pos


@dataclass
class Image:
    path: str
    image: tf.Tensor

    def edges(self) -> Tuple[Edge, Edge, Edge, Edge]:
        height = self.height()
        width = self.width()
        edge_size_w = width // 4
        edge_size_h = height // 4
        return (Edge(self.image[0:edge_size_h, 0:edge_size_w], Pos.TOP_LEFT),
                Edge(self.image[0:edge_size_h, -edge_size_w:], Pos.TOP_RIGHT),
                Edge(self.image[-edge_size_h:, 0:edge_size_w], Pos.BOT_LEFT),
                Edge(self.image[-edge_size_h:, -edge_size_w:], Pos.BOT_RIGHT))

    def height(self) -> int:
        return self.image.shape[0]

    def width(self) -> int:
        return self.image.shape[1]

    def dim(self) -> int:
        return reduce((lambda x, y: x * y), self.image.shape)

    def edges_color_mean(self) -> float:
        edges = self.edges()
        mean_colors = [tf.reduce_mean(edge.val) for edge in edges]
        return tf.reduce_mean(mean_colors).numpy()

    def background_color(self) -> Color:
        edges_mean = self.edges_color_mean()
        threshold = self.threshold_color()
        color = Color.WHITE if edges_mean > threshold else Color.BLACK
        return color

    def threshold_color(self) -> float:
        return THRESHOLD_RATIO * self.image.dtype.max

    def invert_color(self) -> Optional[tf.Tensor]:
        if len(self.image.shape) < 2:
            return None
        back = self.background_color()
        if back == Color.WHITE:
            print(f'inverted {self.path}')
            self.image = self.image.dtype.max - self.image
            return self.image
        return None


class Attribute(Enum):
    NAME = "PatientName"
    INSTITUTION = "InstitutionName"
    STUDY_DATE = "StudyDate"
    ID = "PatientID"
    AGE = "PatientAge"
    SEX = "PatientSex"
    BODY_PART = "BodyPartExamined"


@dataclass
class Patient:
    id: str
    label: str
    images: List[Image] = field(default_factory=list)
    metadata: Dict[Attribute, str] = field(default_factory=dict)
"""
Implementation of the top level class for the ORSO header.
"""

# author: Andrew R. McCluskey (arm61)

from typing import List, Union
from dataclasses import dataclass
from .base import Header, Column, Creator
from .data_source import DataSource
from .reduction import Reduction


@dataclass
class Orso(Header):
    """
    The Orso object collects the necessary metadata.
    """
    creator: Creator
    data_source: Union[DataSource, List[DataSource]]
    reduction: Reduction
    columns: List[Column]
    data_set: Union[str, List[str]]

    __repr__ = Header._staggered_repr

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class IsomorphismMappings:
    mapping: dict

    def __post_init__(self):
        assert self.is_bijective(self.mapping)

    def __getitem__(self, key):
        return self.mapping[key]

    def __iter__(self):
        return self.mapping.__iter__()

    def is_empty(self):
        return self.mapping == {}

    @staticmethod
    def reverse_dictionary(dictionary: dict) -> dict:
        return {value: key for (key, value) in dictionary.items()}

    @staticmethod
    def is_bijective(dictionary: dict) -> bool:
        return len(dictionary) == len(
            IsomorphismMappings.reverse_dictionary(dictionary)
        )

    @property
    def reverse_mapping(self):
        return self.reverse_dictionary(self.mapping)


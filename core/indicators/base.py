"""Base types for modular indicators."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class IndicatorResult:
    name: str
    kind: str
    outputs: dict
    metadata: dict

    def to_frame(self):
        if not self.outputs:
            return pd.DataFrame()
        return pd.DataFrame(self.outputs)


@dataclass
class IndicatorRunResult:
    frame: pd.DataFrame
    results: list

    @property
    def metadata(self):
        return {result.name: result.metadata for result in self.results}


class Indicator:
    """Base class for all indicators."""

    kind = "base"
    required_columns = ()

    def __init__(self, name=None):
        self.name = name or self.default_name()

    def default_name(self):
        return self.kind

    def params(self):
        params = dict(self.__dict__)
        params.pop("name", None)
        return params

    def validate(self, df):
        missing = [column for column in self.required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"{self.kind} requires columns {missing}, got {list(df.columns)}")

    def compute(self, df):
        raise NotImplementedError

    def describe(self, outputs):
        return {
            "kind": self.kind,
            "name": self.name,
            "params": self.params(),
            "required_columns": list(self.required_columns),
            "output_columns": list(outputs),
        }

    def run(self, df):
        self.validate(df)
        outputs = {}
        for column_name, series in self.compute(df).items():
            outputs[column_name] = series.rename(column_name)
        return IndicatorResult(
            name=self.name,
            kind=self.kind,
            outputs=outputs,
            metadata=self.describe(outputs),
        )
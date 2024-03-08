from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionArtifact:
    data_zip_file_path:str
    feature_store_path:str


@dataclass(frozen=True)
class DataValidationArtifact:
    validation_status: bool

@dataclass(frozen=True)
class ModelTrainerArtifact:
    trained_model_file_path: str
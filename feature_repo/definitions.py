from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32

iris_sample = Entity(
    name="iris_sample",
    join_keys=["sample_id"],
    description="A uniquely identified iris flower sample",
)

iris_source = FileSource(
    name="iris_source",
    path="data/iris_features.parquet",
    timestamp_field="event_timestamp",
)

iris_features = FeatureView(
    name="iris_features",
    entities=[iris_sample],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    source=iris_source,
    online=True,
    description="Four numerical features for iris classification",
)

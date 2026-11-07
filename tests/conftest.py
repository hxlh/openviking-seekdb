from __future__ import annotations

import pytest
from openviking.storage.collection_schemas import CollectionSchemas


@pytest.fixture
def schema_dict():
    return CollectionSchemas.context_collection("context", 4, description="test collection")

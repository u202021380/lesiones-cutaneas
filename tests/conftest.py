import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Las pruebas automáticas usan el modelo simulado salvo que se pida el real
os.environ.setdefault("USAR_MODELO_SIMULADO", "1")


@pytest.fixture(scope="session")
def cliente():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)

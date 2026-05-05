import pytest
from app.services.extractor import parse_dimension, parse_pressure_class
from app.models.domain import PressureClass

@pytest.mark.parametrize("text, expected_dims", [
    ('14"⌀',        (0, 0, 14.0)),
    ('14 inch dia',  (0, 0, 14.0)),
    ('10x8',         (10.0, 8.0, 0)),
    ('10 X 8',       (10.0, 8.0, 0)),
    ('10"x8"',       (10.0, 8.0, 0)),
    ('12"×10" LP',   (12.0, 10.0, 0)),
    ('14x10 HP',     (14.0, 10.0, 0)),
    ('Return Air 12"', None),
    ('room 101',     None),
    ('SUPPLY 14x10 MP', (14.0, 10.0, 0)),
    ('invalid text', None),
])
def test_parse_dimension(text, expected_dims):
    assert parse_dimension(text) == expected_dims

@pytest.mark.parametrize("text, expected_pressure", [
    ('12"×10" LP',   PressureClass.LOW),
    ('14x10 HP',     PressureClass.HIGH),
    ('SUPPLY 14x10 MP', PressureClass.MEDIUM),
    ('Return Air 12"', PressureClass.UNKNOWN),
])
def test_parse_pressure_class(text, expected_pressure):
    assert parse_pressure_class(text) == expected_pressure

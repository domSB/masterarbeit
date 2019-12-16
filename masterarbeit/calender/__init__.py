# -*- coding: utf-8 -*-
"""
Copyright (C) 2015 Davis Kirkendall
siehe LICENCE File

Ein Pandas Kalender für die deutschen Feiertage,
insbesondere mit Berücksichtigung der regionalen Unterschiede.
"""
from calender.german_holidays import get_german_holiday_calendar  # noqa: F401

__all__ = ['german_holidays', 'state_codes']

# FPL Team Picker

Fantasy Premier League analysis suite providing season-start team building and weekly gameweek management through comprehensive mathematical modeling.

## Overview

Two complementary optimization approaches:
- **Season Planning**: Multi-gameweek team building with 5-week horizon optimization
- **Weekly Management**: Form-weighted gameweek decisions with live data integration

## Installation

```bash
uv sync
```

This automatically installs the local `fpl-dataset-builder` dependency and all required packages.

## Usage

**Season-start team building:**
```bash
marimo run fpl_xp_model.py
```

**Weekly gameweek management:**
```bash
marimo run fpl_gameweek_manager.py
```

## Data Source

Uses the `fpl-dataset-builder` database client for fresh data from a centralized database. No CSV dependencies - all data loaded via:

```python
from client import get_current_players, get_fixtures_normalized, get_gameweek_live_data
```

The fpl-dataset-builder is configured as a local editable dependency in `pyproject.toml`.
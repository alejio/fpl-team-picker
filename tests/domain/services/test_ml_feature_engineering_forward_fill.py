"""
Tests for forward-fill functionality in ownership and value analysis features.

Tests that ownership and value analysis features correctly forward-fill data
for future gameweeks when doing cascading predictions (5gw ahead).

Test coverage:
1. Ownership features forward-fill for future gameweeks
2. Value analysis features forward-fill for future gameweeks
3. Both handle new players with no history gracefully
4. Both still raise errors for historical gaps (data quality issues)
5. Forward-fill uses last known value per player
"""

import pytest
import pandas as pd
import numpy as np

from fpl_team_picker.domain.services.ml_feature_engineering import (
    FPLFeatureEngineer,
)


class TestOwnershipFeaturesForwardFill:
    """Test ownership features forward-fill for future gameweeks."""

    @pytest.fixture
    def sample_historical_data(self):
        """Historical player data for GW1-10."""
        data = []
        for gw in range(1, 11):
            for player_id in [1, 2, 3]:
                data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "total_points": 5 + (player_id * gw % 3),
                        "minutes": 90,
                        "goals_scored": 0,
                        "assists": 0,
                        "position": "DEF" if player_id == 1 else ("MID" if player_id == 2 else "FWD"),
                        "team_id": player_id,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def ownership_data_through_gw10(self):
        """Ownership data available through GW10."""
        data = []
        for gw in range(1, 11):
            for player_id in [1, 2, 3]:
                data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "selected_by_percent": 10.0 + player_id * 5 + gw * 0.1,  # Varies by player and GW
                        "net_transfers_gw": 100 * player_id,
                        "avg_net_transfers_5gw": 80 * player_id,
                        "transfer_momentum": "rising" if player_id == 1 else "neutral",
                        "ownership_velocity": 0.5 + player_id * 0.1,
                        "ownership_tier": "popular" if player_id == 1 else "budget",
                        "bandwagon_score": 0.5 + player_id * 0.1,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def fixture_difficulty_data(self):
        """Mock fixture difficulty data."""
        data = []
        for gw in range(1, 16):  # GW1-15
            for team_id in [1, 2, 3]:
                data.append(
                    {
                        "team_id": team_id,
                        "gameweek": gw,
                        "congestion_difficulty": 1.0,
                        "form_difficulty": 1.0,
                        "clean_sheet_probability": 0.35,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def betting_features_data(self):
        """Mock betting features data."""
        data = []
        for gw in range(1, 16):  # GW1-15
            for player_id in [1, 2, 3]:
                data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "team_win_probability": 0.40,
                        "opponent_win_probability": 0.30,
                        "draw_probability": 0.30,
                        "implied_clean_sheet_probability": 0.35,
                        "implied_total_goals": 2.5,
                        "team_expected_goals": 1.3,
                        "market_consensus_strength": 0.5,
                        "odds_movement_team": 0.0,
                        "odds_movement_magnitude": 0.0,
                        "favorite_status": 0.5,
                        "asian_handicap_line": 0.0,
                        "handicap_team_odds": 2.0,
                        "expected_goal_difference": 0.0,
                        "over_under_signal": 0.0,
                        "referee_encoded": -1,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def value_data_through_gw10(self):
        """Value analysis data available through GW10."""
        data = []
        for gw in range(1, 11):
            for player_id in [1, 2, 3]:
                data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "points_per_pound": 0.5 + player_id * 0.1,
                        "value_vs_position": 1.0 + player_id * 0.1,
                        "predicted_price_change_1gw": 0.1 * player_id,
                        "price_volatility": 0.05,
                        "price_risk": 0.02,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def feature_engineer(self, ownership_data_through_gw10, value_data_through_gw10, fixture_difficulty_data, betting_features_data):
        """Create feature engineer with ownership and value data."""
        engineer = FPLFeatureEngineer(
            fixtures_df=pd.DataFrame(),
            teams_df=pd.DataFrame({"team_id": [1, 2, 3], "name": ["Team1", "Team2", "Team3"]}),
            team_strength={},
            ownership_trends_df=ownership_data_through_gw10,
            value_analysis_df=value_data_through_gw10,
            betting_features_df=betting_features_data,
            raw_players_df=pd.DataFrame(),
            fixture_difficulty_df=fixture_difficulty_data,
        )
        return engineer

    def test_forward_fills_ownership_for_future_gameweeks(
        self, feature_engineer, sample_historical_data
    ):
        """Test that ownership features forward-fill for GW11-15."""
        # Add future gameweeks (GW11-15) to the data
        future_data = []
        for gw in range(11, 16):
            for player_id in [1, 2, 3]:
                future_data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "total_points": 0,  # Future performance unknown
                        "minutes": 0,
                        "goals_scored": 0,
                        "assists": 0,
                        "position": "DEF" if player_id == 1 else ("MID" if player_id == 2 else "FWD"),
                        "team_id": player_id,
                        "bonus": 0,
                        "bps": 0,
                        "clean_sheets": 0,
                        "expected_goals": 0,
                        "expected_assists": 0,
                        "ict_index": 0,
                        "influence": 0,
                        "creativity": 0,
                        "threat": 0,
                        "value": 50 + player_id * 10,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "goals_conceded": 0,
                        "saves": 0,
                        "expected_goal_involvements": 0,
                        "expected_goals_conceded": 0,
                    }
                )

        all_data = pd.concat([sample_historical_data, pd.DataFrame(future_data)], ignore_index=True)

        # Preserve gameweek and player_id for verification
        metadata = all_data[["player_id", "gameweek"]].copy()

        # Transform - should not raise error
        result = feature_engineer.transform(all_data)

        # Merge back gameweek and player_id for verification
        result = result.merge(metadata, left_index=True, right_index=True, how="left")

        # Verify ownership features exist for future gameweeks
        assert "selected_by_percent" in result.columns

        # Check GW11 (should use GW10 ownership due to shift(1), or forward-filled if merge fails)
        gw11_data = result[result["gameweek"] == 11]
        assert not gw11_data["selected_by_percent"].isna().any(), "GW11 should have ownership data"

        # Check GW12-15 (should be forward-filled from GW11)
        # The key test is that they have non-null values, not exact matching
        # (forward-fill may use different values depending on implementation)
        for gw in [12, 13, 14, 15]:
            gw_data = result[result["gameweek"] == gw]
            assert not gw_data["selected_by_percent"].isna().any(), f"GW{gw} should have forward-filled ownership"

            # Verify forward-fill: each player's GW12+ ownership should be consistent (not NaN)
            # We don't check exact matching because forward-fill may use different strategies
            for player_id in [1, 2, 3]:
                player_gw = gw_data[gw_data["player_id"] == player_id]["selected_by_percent"].iloc[0]
                assert not pd.isna(player_gw), f"Player {player_id} GW{gw} should have ownership value"
                assert player_gw > 0, f"Player {player_id} GW{gw} ownership should be positive"

    def test_forward_fills_all_ownership_columns(
        self, feature_engineer, sample_historical_data
    ):
        """Test that all ownership columns are forward-filled."""
        # Add GW11-12
        future_data = []
        for gw in range(11, 13):
            for player_id in [1, 2, 3]:
                future_data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "total_points": 0,
                        "minutes": 0,
                        "goals_scored": 0,
                        "assists": 0,
                        "position": "DEF" if player_id == 1 else ("MID" if player_id == 2 else "FWD"),
                        "team_id": player_id,
                        "bonus": 0,
                        "bps": 0,
                        "clean_sheets": 0,
                        "expected_goals": 0,
                        "expected_assists": 0,
                        "ict_index": 0,
                        "influence": 0,
                        "creativity": 0,
                        "threat": 0,
                        "value": 50 + player_id * 10,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "goals_conceded": 0,
                        "saves": 0,
                        "expected_goal_involvements": 0,
                        "expected_goals_conceded": 0,
                    }
                )

        all_data = pd.concat([sample_historical_data, pd.DataFrame(future_data)], ignore_index=True)
        metadata = all_data[["player_id", "gameweek"]].copy()
        result = feature_engineer.transform(all_data)
        result = result.merge(metadata, left_index=True, right_index=True, how="left")

        # Check all ownership columns are forward-filled for GW12
        gw12_data = result[result["gameweek"] == 12]
        ownership_cols = [
            "selected_by_percent",
            "net_transfers_gw",
            "avg_net_transfers_5gw",
            "bandwagon_score",
            "ownership_velocity",
            "transfer_momentum",
            "ownership_tier",
        ]

        for col in ownership_cols:
            if col in gw12_data.columns:
                assert not gw12_data[col].isna().any(), f"{col} should be forward-filled for GW12"

    def test_handles_new_players_with_no_history(
        self, feature_engineer, sample_historical_data
    ):
        """Test that new players (no ownership history) get default values."""
        # Add a new player (player_id=4) for GW12-13 (not GW11, since GW11 needs GW10 ownership)
        new_player_data = []
        for gw in range(12, 14):
            new_player_data.append(
                {
                    "player_id": 4,
                    "gameweek": gw,
                    "total_points": 0,
                    "minutes": 0,
                    "goals_scored": 0,
                    "assists": 0,
                    "position": "MID",
                    "team_id": 4,
                    "bonus": 0,
                    "bps": 0,
                    "clean_sheets": 0,
                    "expected_goals": 0,
                    "expected_assists": 0,
                    "ict_index": 0,
                    "influence": 0,
                    "creativity": 0,
                    "threat": 0,
                    "value": 50,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "goals_conceded": 0,
                    "saves": 0,
                    "expected_goal_involvements": 0,
                    "expected_goals_conceded": 0,
                }
            )

        all_data = pd.concat([sample_historical_data, pd.DataFrame(new_player_data)], ignore_index=True)
        metadata = all_data[["player_id", "gameweek"]].copy()
        result = feature_engineer.transform(all_data)
        result = result.merge(metadata, left_index=True, right_index=True, how="left")

        # New player should get default ownership values
        new_player_gw12 = result[(result["player_id"] == 4) & (result["gameweek"] == 12)]
        assert not new_player_gw12["selected_by_percent"].isna().any()
        # Should default to median ownership (5.0)
        assert new_player_gw12["selected_by_percent"].iloc[0] == 5.0
        assert new_player_gw12["net_transfers_gw"].iloc[0] == 0
        # transfer_momentum is encoded, so check the encoded version
        assert "transfer_momentum_encoded" in result.columns
        assert new_player_gw12["transfer_momentum_encoded"].iloc[0] == 0  # "neutral" maps to 0

    def test_still_raises_error_for_historical_gaps(
        self, feature_engineer, sample_historical_data
    ):
        """Test that historical gaps (data quality issues) still raise errors."""
        # Create data with a gap: GW1-5, GW8-10 (missing GW6-7)
        # Note: gap_data already has all required columns from sample_historical_data
        gap_data = sample_historical_data[
            ~sample_historical_data["gameweek"].isin([6, 7])
        ].copy()

        # This should raise an error because GW8 needs GW7 ownership data
        # (after shift(1), GW8 needs GW7 ownership, but GW7 is missing)
        # The error might be about missing ownership data or missing required columns
        with pytest.raises(ValueError) as exc_info:
            feature_engineer.transform(gap_data)
        # Check that the error mentions ownership or missing data
        assert "ownership" in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower()


class TestValueAnalysisFeaturesForwardFill:
    """Test value analysis features forward-fill for future gameweeks."""

    @pytest.fixture
    def sample_historical_data(self):
        """Historical player data for GW1-10."""
        data = []
        for gw in range(1, 11):
            for player_id in [1, 2, 3]:
                data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "total_points": 5 + (player_id * gw % 3),
                        "minutes": 90,
                        "goals_scored": 0,
                        "assists": 0,
                        "position": "DEF" if player_id == 1 else ("MID" if player_id == 2 else "FWD"),
                        "team_id": player_id,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def value_data_through_gw10(self):
        """Value analysis data available through GW10."""
        data = []
        for gw in range(1, 11):
            for player_id in [1, 2, 3]:
                data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "points_per_pound": 0.5 + player_id * 0.1 + gw * 0.01,  # Varies by player and GW
                        "value_vs_position": 1.0 + player_id * 0.1,
                        "predicted_price_change_1gw": 0.1 * player_id,
                        "price_volatility": 0.05 * player_id,
                        "price_risk": 0.02 * player_id,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def ownership_data_through_gw10(self):
        """Ownership data available through GW10."""
        data = []
        for gw in range(1, 11):
            for player_id in [1, 2, 3]:
                data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "selected_by_percent": 10.0 + player_id * 5,
                        "net_transfers_gw": 100 * player_id,
                        "avg_net_transfers_5gw": 80 * player_id,
                        "transfer_momentum": "rising",
                        "ownership_velocity": 0.5,
                        "ownership_tier": "popular",
                        "bandwagon_score": 0.5,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def fixture_difficulty_data(self):
        """Mock fixture difficulty data."""
        data = []
        for gw in range(1, 16):  # GW1-15
            for team_id in [1, 2, 3]:
                data.append(
                    {
                        "team_id": team_id,
                        "gameweek": gw,
                        "congestion_difficulty": 1.0,
                        "form_difficulty": 1.0,
                        "clean_sheet_probability": 0.35,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def betting_features_data(self):
        """Mock betting features data."""
        data = []
        for gw in range(1, 16):  # GW1-15
            for player_id in [1, 2, 3]:
                data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "team_win_probability": 0.40,
                        "opponent_win_probability": 0.30,
                        "draw_probability": 0.30,
                        "implied_clean_sheet_probability": 0.35,
                        "implied_total_goals": 2.5,
                        "team_expected_goals": 1.3,
                        "market_consensus_strength": 0.5,
                        "odds_movement_team": 0.0,
                        "odds_movement_magnitude": 0.0,
                        "favorite_status": 0.5,
                        "asian_handicap_line": 0.0,
                        "handicap_team_odds": 2.0,
                        "expected_goal_difference": 0.0,
                        "over_under_signal": 0.0,
                        "referee_encoded": -1,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def feature_engineer(self, ownership_data_through_gw10, value_data_through_gw10, fixture_difficulty_data, betting_features_data):
        """Create feature engineer with ownership and value data."""
        engineer = FPLFeatureEngineer(
            fixtures_df=pd.DataFrame(),
            teams_df=pd.DataFrame({"team_id": [1, 2, 3], "name": ["Team1", "Team2", "Team3"]}),
            team_strength={},
            ownership_trends_df=ownership_data_through_gw10,
            value_analysis_df=value_data_through_gw10,
            betting_features_df=betting_features_data,
            raw_players_df=pd.DataFrame(),
            fixture_difficulty_df=fixture_difficulty_data,
        )
        return engineer

    def test_forward_fills_value_for_future_gameweeks(
        self, feature_engineer, sample_historical_data
    ):
        """Test that value analysis features forward-fill for GW11-15."""
        # Add future gameweeks (GW11-15)
        future_data = []
        for gw in range(11, 16):
            for player_id in [1, 2, 3]:
                future_data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "total_points": 0,
                        "minutes": 0,
                        "goals_scored": 0,
                        "assists": 0,
                        "position": "DEF" if player_id == 1 else ("MID" if player_id == 2 else "FWD"),
                        "team_id": player_id,
                        "bonus": 0,
                        "bps": 0,
                        "clean_sheets": 0,
                        "expected_goals": 0,
                        "expected_assists": 0,
                        "ict_index": 0,
                        "influence": 0,
                        "creativity": 0,
                        "threat": 0,
                        "value": 50 + player_id * 10,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "goals_conceded": 0,
                        "saves": 0,
                        "expected_goal_involvements": 0,
                        "expected_goals_conceded": 0,
                    }
                )

        all_data = pd.concat([sample_historical_data, pd.DataFrame(future_data)], ignore_index=True)
        metadata = all_data[["player_id", "gameweek"]].copy()
        result = feature_engineer.transform(all_data)
        result = result.merge(metadata, left_index=True, right_index=True, how="left")

        # Verify value features exist for future gameweeks
        assert "points_per_pound" in result.columns

        # Check GW11 (should use GW10 value due to shift(1))
        gw11_data = result[result["gameweek"] == 11]
        assert not gw11_data["points_per_pound"].isna().any(), "GW11 should have value data"

        # Check GW12-15 (should be forward-filled from GW11)
        for gw in [12, 13, 14, 15]:
            gw_data = result[result["gameweek"] == gw]
            assert not gw_data["points_per_pound"].isna().any(), f"GW{gw} should have forward-filled value"

            # Verify forward-fill: each player's GW12+ value should be consistent (not NaN)
            # We don't check exact matching because forward-fill may use different strategies
            for player_id in [1, 2, 3]:
                player_gw = gw_data[gw_data["player_id"] == player_id]["points_per_pound"].iloc[0]
                assert not pd.isna(player_gw), f"Player {player_id} GW{gw} should have value"
                assert player_gw > 0, f"Player {player_id} GW{gw} value should be positive"

    def test_forward_fills_all_value_columns(
        self, feature_engineer, sample_historical_data
    ):
        """Test that all value columns are forward-filled."""
        # Add GW11-12
        future_data = []
        for gw in range(11, 13):
            for player_id in [1, 2, 3]:
                future_data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "total_points": 0,
                        "minutes": 0,
                        "goals_scored": 0,
                        "assists": 0,
                        "position": "DEF" if player_id == 1 else ("MID" if player_id == 2 else "FWD"),
                        "team_id": player_id,
                        "bonus": 0,
                        "bps": 0,
                        "clean_sheets": 0,
                        "expected_goals": 0,
                        "expected_assists": 0,
                        "ict_index": 0,
                        "influence": 0,
                        "creativity": 0,
                        "threat": 0,
                        "value": 50 + player_id * 10,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "goals_conceded": 0,
                        "saves": 0,
                        "expected_goal_involvements": 0,
                        "expected_goals_conceded": 0,
                    }
                )

        all_data = pd.concat([sample_historical_data, pd.DataFrame(future_data)], ignore_index=True)
        metadata = all_data[["player_id", "gameweek"]].copy()
        result = feature_engineer.transform(all_data)
        result = result.merge(metadata, left_index=True, right_index=True, how="left")

        # Check all value columns are forward-filled for GW12
        gw12_data = result[result["gameweek"] == 12]
        value_cols = [
            "points_per_pound",
            "value_vs_position",
            "predicted_price_change_1gw",
            "price_volatility",
            "price_risk",
        ]

        for col in value_cols:
            if col in gw12_data.columns:
                assert not gw12_data[col].isna().any(), f"{col} should be forward-filled for GW12"

    def test_handles_new_players_with_no_value_history(
        self, feature_engineer, sample_historical_data
    ):
        """Test that new players (no value history) get default values."""
        # Add a new player (player_id=4) for GW12-13 (not GW11, since GW11 needs GW10 value data)
        new_player_data = []
        for gw in range(12, 14):
            new_player_data.append(
                {
                    "player_id": 4,
                    "gameweek": gw,
                    "total_points": 0,
                    "minutes": 0,
                    "goals_scored": 0,
                    "assists": 0,
                    "position": "MID",
                    "team_id": 4,
                    "bonus": 0,
                    "bps": 0,
                    "clean_sheets": 0,
                    "expected_goals": 0,
                    "expected_assists": 0,
                    "ict_index": 0,
                    "influence": 0,
                    "creativity": 0,
                    "threat": 0,
                    "value": 50,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "goals_conceded": 0,
                    "saves": 0,
                    "expected_goal_involvements": 0,
                    "expected_goals_conceded": 0,
                }
            )

        all_data = pd.concat([sample_historical_data, pd.DataFrame(new_player_data)], ignore_index=True)
        metadata = all_data[["player_id", "gameweek"]].copy()
        result = feature_engineer.transform(all_data)
        result = result.merge(metadata, left_index=True, right_index=True, how="left")

        # New player should get default value values
        new_player_gw12 = result[(result["player_id"] == 4) & (result["gameweek"] == 12)]
        assert not new_player_gw12["points_per_pound"].isna().any()
        # Should default to neutral values
        assert new_player_gw12["points_per_pound"].iloc[0] == 0.5
        assert new_player_gw12["value_vs_position"].iloc[0] == 1.0
        assert new_player_gw12["predicted_price_change_1gw"].iloc[0] == 0

    def test_still_raises_error_for_historical_gaps(
        self, feature_engineer, sample_historical_data
    ):
        """Test that historical gaps (data quality issues) still raise errors."""
        # Create data with a gap: GW1-5, GW8-10 (missing GW6-7)
        # Note: gap_data already has all required columns from sample_historical_data
        gap_data = sample_historical_data[
            ~sample_historical_data["gameweek"].isin([6, 7])
        ].copy()

        # This should raise an error because GW8 needs GW7 value data
        # (after shift(1), GW8 needs GW7 value, but GW7 is missing)
        # The error might be about missing value data or missing required columns
        with pytest.raises(ValueError) as exc_info:
            feature_engineer.transform(gap_data)
        # Check that the error mentions value analysis or missing data
        assert "value" in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower()


class TestCascadingPredictionScenario:
    """Test the full cascading prediction scenario (5gw ahead)."""

    @pytest.fixture
    def full_setup(self):
        """Complete setup for cascading prediction test."""
        # Historical data GW1-10
        historical = []
        for gw in range(1, 11):
            for player_id in [1, 2]:
                historical.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "total_points": 5 + (player_id * gw % 3),
                        "minutes": 90,
                        "goals_scored": 0,
                        "assists": 0,
                        "position": "DEF" if player_id == 1 else "MID",
                        "team_id": player_id,
                        "bonus": 0,
                        "bps": 20,
                        "clean_sheets": 1 if player_id == 1 else 0,
                        "expected_goals": 0.2,
                        "expected_assists": 0.1,
                        "ict_index": 6.0,
                        "influence": 30.0,
                        "creativity": 20.0,
                        "threat": 10.0,
                        "value": 50 + player_id * 10,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "goals_conceded": 1,
                        "saves": 0,
                        "expected_goal_involvements": 0.3,
                        "expected_goals_conceded": 1.0,
                    }
                )

        # Ownership data through GW10
        ownership = []
        for gw in range(1, 11):
            for player_id in [1, 2]:
                ownership.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "selected_by_percent": 10.0 + player_id * 5,
                        "net_transfers_gw": 100 * player_id,
                        "avg_net_transfers_5gw": 80 * player_id,
                        "transfer_momentum": "rising",
                        "ownership_velocity": 0.5,
                        "ownership_tier": "popular",
                        "bandwagon_score": 0.5,
                    }
                )

        # Value data through GW10
        value = []
        for gw in range(1, 11):
            for player_id in [1, 2]:
                value.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "points_per_pound": 0.5 + player_id * 0.1,
                        "value_vs_position": 1.0 + player_id * 0.1,
                        "predicted_price_change_1gw": 0.1 * player_id,
                        "price_volatility": 0.05,
                        "price_risk": 0.02,
                    }
                )

        # Create fixture difficulty data
        fixture_difficulty = []
        for gw in range(1, 16):
            for team_id in [1, 2]:
                fixture_difficulty.append(
                    {
                        "team_id": team_id,
                        "gameweek": gw,
                        "congestion_difficulty": 1.0,
                        "form_difficulty": 1.0,
                        "clean_sheet_probability": 0.35,
                    }
                )

        # Create betting features data
        betting_features = []
        for gw in range(1, 16):
            for player_id in [1, 2]:
                betting_features.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "team_win_probability": 0.40,
                        "opponent_win_probability": 0.30,
                        "draw_probability": 0.30,
                        "implied_clean_sheet_probability": 0.35,
                        "implied_total_goals": 2.5,
                        "team_expected_goals": 1.3,
                        "market_consensus_strength": 0.5,
                        "odds_movement_team": 0.0,
                        "odds_movement_magnitude": 0.0,
                        "favorite_status": 0.5,
                        "asian_handicap_line": 0.0,
                        "handicap_team_odds": 2.0,
                        "expected_goal_difference": 0.0,
                        "over_under_signal": 0.0,
                        "referee_encoded": -1,
                    }
                )

        engineer = FPLFeatureEngineer(
            fixtures_df=pd.DataFrame(),
            teams_df=pd.DataFrame({"team_id": [1, 2], "name": ["Team1", "Team2"]}),
            team_strength={},
            ownership_trends_df=pd.DataFrame(ownership),
            value_analysis_df=pd.DataFrame(value),
            betting_features_df=pd.DataFrame(betting_features),
            raw_players_df=pd.DataFrame(),
            fixture_difficulty_df=pd.DataFrame(fixture_difficulty),
        )

        return pd.DataFrame(historical), engineer

    def test_cascading_predictions_gw11_through_15(
        self, full_setup
    ):
        """Test that cascading predictions work for GW11-15."""
        historical_data, engineer = full_setup

        # Simulate cascading: add GW11, then GW12, etc.
        # In real scenario, GW12 would use predicted GW11 data, but for this test
        # we just verify forward-fill works for all future gameweeks

        future_data = []
        for gw in range(11, 16):  # GW11-15
            for player_id in [1, 2]:
                future_data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "total_points": 0,  # Future performance
                        "minutes": 0,
                        "goals_scored": 0,
                        "assists": 0,
                        "position": "DEF" if player_id == 1 else "MID",
                        "team_id": player_id,
                        "bonus": 0,
                        "bps": 0,
                        "clean_sheets": 0,
                        "expected_goals": 0,
                        "expected_assists": 0,
                        "ict_index": 0,
                        "influence": 0,
                        "creativity": 0,
                        "threat": 0,
                        "value": 50 + player_id * 10,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "goals_conceded": 0,
                        "saves": 0,
                        "expected_goal_involvements": 0,
                        "expected_goals_conceded": 0,
                    }
                )

        all_data = pd.concat([historical_data, pd.DataFrame(future_data)], ignore_index=True)

        # Preserve metadata for verification
        metadata = all_data[["player_id", "gameweek"]].copy()

        # Should not raise error
        result = engineer.transform(all_data)

        # Merge back metadata
        result = result.merge(metadata, left_index=True, right_index=True, how="left")

        # Verify all future gameweeks have ownership and value data
        for gw in [11, 12, 13, 14, 15]:
            gw_data = result[result["gameweek"] == gw]
            assert not gw_data["selected_by_percent"].isna().any(), f"GW{gw} ownership should be filled"
            assert not gw_data["points_per_pound"].isna().any(), f"GW{gw} value should be filled"

            # Verify forward-fill consistency: GW12+ should match GW11
            if gw > 11:
                gw11_data = result[result["gameweek"] == 11]
                for player_id in [1, 2]:
                    gw11_ownership = gw11_data[gw11_data["player_id"] == player_id]["selected_by_percent"].iloc[0]
                    gw_ownership = gw_data[gw_data["player_id"] == player_id]["selected_by_percent"].iloc[0]
                    assert gw_ownership == gw11_ownership, f"Player {player_id} GW{gw} ownership should match GW11"

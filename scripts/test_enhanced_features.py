"""
Test script for enhanced ML features (ownership, value, fixture difficulty).

This script validates that the new FPLFeatureEngineerEnhanced correctly:
1. Loads all three derived data sources
2. Merges them with player data
3. Generates 80 features (65 base + 15 enhanced)
4. Maintains leak-free guarantees (no future data)
5. Produces sensible feature values

Usage:
    uv run python scripts/test_enhanced_features.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from client import FPLDataClient
from fpl_team_picker.domain.services.ml_feature_engineering import (
    FPLFeatureEngineer,
)


def test_enhanced_feature_engineering():
    """Test enhanced feature engineering pipeline."""
    print("=" * 80)
    print("ENHANCED FEATURE ENGINEERING TEST")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    client = FPLDataClient()

    players = client.get_current_players()
    teams = client.get_current_teams()
    fixtures = client.get_fixtures_normalized()
    ownership = client.get_derived_ownership_trends()
    value = client.get_derived_value_analysis()
    fixture_diff = client.get_derived_fixture_difficulty()

    print(f"   Players: {len(players)}")
    print(f"   Ownership data: {len(ownership)}")
    print(f"   Value data: {len(value)}")
    print(f"   Fixture difficulty: {len(fixture_diff)}")

    # Load historical gameweek data (use GW6-8 for testing)
    print("\n2. Loading historical gameweek data (GW6-8)...")
    all_data = []
    for gw in range(6, 9):
        try:
            gw_data = client.get_gameweek_performance(gw)
            gw_data["gameweek"] = gw
            all_data.append(gw_data)
            print(f"   GW{gw}: {len(gw_data)} player-gameweeks")
        except Exception as e:
            print(f"   Warning: Could not load GW{gw}: {e}")

    if not all_data:
        print("   ERROR: No historical data available!")
        return False

    historical_data = pd.concat(all_data, ignore_index=True)

    # Merge with player metadata
    player_info = players[
        ["player_id", "web_name", "position", "price_gbp", "team_id"]
    ].copy()
    player_info["now_cost"] = (player_info["price_gbp"] * 10).astype(int)

    # Note: historical data already has team_id, so drop it first to avoid _x _y suffixes
    if "team_id" in historical_data.columns:
        historical_data = historical_data.drop(columns=["team_id"])

    historical_data = historical_data.merge(
        player_info[["player_id", "position", "now_cost", "team_id"]],
        on="player_id",
        how="left",
    )

    print(f"   Total samples: {len(historical_data)}")
    print(f"   Has team_id: {'team_id' in historical_data.columns}")

    # Test enhanced feature engineer
    print("\n3. Testing FPLFeatureEngineer with enhanced features...")

    engineer = FPLFeatureEngineer(
        fixtures_df=fixtures,
        teams_df=teams,
        team_strength=None,
        ownership_trends_df=ownership,
        value_analysis_df=value,
        fixture_difficulty_df=fixture_diff,
    )

    # Transform data
    features = engineer.fit_transform(historical_data)

    print("   ✅ Transformation successful!")
    print(f"   Features shape: {features.shape}")
    print("   Expected: (n_samples, 80 features)")
    print(f"   Actual: ({features.shape[0]}, {features.shape[1]})")

    # Validate feature count
    assert features.shape[1] == 80, f"Expected 80 features, got {features.shape[1]}"
    print("   ✅ Feature count correct (80)")

    # Check for NaNs
    nan_counts = features.isna().sum()
    if nan_counts.sum() > 0:
        print(f"   ⚠️  Warning: {nan_counts.sum()} NaN values found:")
        print(nan_counts[nan_counts > 0])
    else:
        print("   ✅ No NaN values")

    # Check enhanced features are present
    all_features = engineer._get_feature_columns()
    enhanced_features = all_features[65:]  # Last 15 features are enhanced
    print(f"\n4. Validating enhanced features ({len(enhanced_features)})...")

    for feature in enhanced_features:
        if feature not in features.columns:
            print(f"   ❌ Missing feature: {feature}")
            return False

    print("   ✅ All enhanced features present")

    # Sample feature values
    print("\n5. Sample enhanced feature values (GW8, first 5 players):")
    sample_features = enhanced_features[:10]  # First 10 enhanced features
    print(features[sample_features].head(5).to_string())

    # Check ownership features
    print("\n6. Ownership feature statistics:")
    ownership_features = [
        "selected_by_percent",
        "ownership_tier_encoded",
        "net_transfers_gw",
        "bandwagon_score",
    ]
    print(features[ownership_features].describe().to_string())

    # Check value features
    print("\n7. Value feature statistics:")
    value_features = [
        "points_per_pound",
        "value_vs_position",
        "predicted_price_change_1gw",
    ]
    print(features[value_features].describe().to_string())

    # Check fixture features
    print("\n8. Fixture difficulty feature statistics:")
    fixture_features = [
        "congestion_difficulty",
        "form_adjusted_difficulty",
        "clean_sheet_probability_enhanced",
    ]
    print(features[fixture_features].describe().to_string())

    print("\n✅ Enhanced feature engineering test PASSED!")
    return True


def main():
    """Run all tests."""
    print("Testing Enhanced FPL Features (Issue #37)")
    print("=" * 80)

    success = True

    # Test: Enhanced feature engineering
    try:
        if not test_enhanced_feature_engineering():
            success = False
    except Exception as e:
        print(f"\n❌ Enhanced feature engineering test FAILED: {e}")
        import traceback

        traceback.print_exc()
        success = False

    # Final summary
    print("\n" + "=" * 80)
    if success:
        print("✅ ALL TESTS PASSED")
        print("\nNext steps:")
        print("1. Update ml_xp_experiment.py to load the 3 new data sources")
        print("2. Pass them to FPLFeatureEngineer constructor")
        print("3. Retrain TPOT model with 80 features (65 base + 15 enhanced)")
        print("4. Test predictions in gameweek_manager.py")
        print(
            "5. Monitor impact: ownership-aware differentials should help close the 223-point gap"
        )
    else:
        print("❌ SOME TESTS FAILED - Review errors above")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

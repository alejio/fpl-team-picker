# Missing Data Sources: Strategic Analysis

**Context**: Currently 223 points behind FPL #1 (ZayceCFC) after 8 gameweeks (~28 pts/GW deficit)

**Key Question**: What data sources could reveal player selection opportunities we're currently missing?

---

## Executive Summary

Based on analysis of your fpl-dataset-builder and fpl-team-picker codebase, you have **5 unused high-value data sources already available** and **3 external sources** worth investigating.

### Quick Wins (Data Already Available, Not Used)
1. ✅ **Ownership trends** - Find differentials (low ownership, high xP)
2. ✅ **Value analysis** - Price changes, value picks
3. ✅ **Fixture difficulty** - Enhanced fixture analysis

### High-Impact External Sources (Worth Adding)
1. 🎯 **Top 10k manager decisions** - What do elite players pick?
2. 🎯 **Bookmaker odds** - Market-implied probabilities
3. 🎯 **Press conference data** - Rotation/injury hints

---

## Data Audit: What You Have vs What You Use

### ✅ Available Data Sources (fpl-dataset-builder)

| Data Source | Available? | Currently Used? | Impact Potential |
|-------------|-----------|-----------------|------------------|
| Historical performance (xG, xA, points) | ✅ | ✅ | High - Core xP model |
| Team strength / form | ✅ | ✅ | High - Already integrated |
| Fixtures | ✅ | ✅ | Medium - Basic usage |
| Player prices | ✅ | ✅ | High - Constraint only |
| Set piece takers | ✅ | ✅ | Medium - Bonus pts |
| **Ownership trends** | ✅ | ❌ **UNUSED** | **HIGH** |
| **Derived value analysis** | ✅ | ❌ **UNUSED** | **MEDIUM** |
| **Derived fixture difficulty** | ✅ | ❌ **UNUSED** | **MEDIUM** |
| Player availability snapshots | ✅ | ✅ | High - Historical accuracy |
| My picks history | ✅ | ✅ | Low - Self-reference |

### ❌ Missing External Data Sources

| Data Source | Difficulty | Impact | Priority |
|-------------|-----------|--------|----------|
| Top 10k ownership | Medium | **Very High** | **P0** |
| Bookmaker odds | Easy | **High** | **P0** |
| Press conferences | Hard | Medium | P2 |
| Detailed fixture congestion | Easy | Medium | P1 |
| Referee statistics | Medium | Low-Medium | P2 |
| Weather data | Medium | Low | P3 |

---

## Detailed Analysis: Missing Data Sources

### 1. Ownership Trends (Available, Not Used) ⭐⭐⭐

**What it is**: `get_derived_ownership_trends()` - Transfer momentum, ownership patterns

**Why it matters**:
- **Differential strategy**: Low ownership + high xP = huge rank gains when they haul
- **Template avoidance**: Everyone owns Salah - if he blanks, no relative loss
- **Transfer trends**: If 500k managers are selling a player, why? Inside info?

**Example opportunity**:
```
Palmer: 45% owned, 9.5 xP next GW
Haaland: 85% owned, 10 xP next GW

→ Palmer differential: If he gets 15 pts, you gain on 55% of managers
→ Haaland template: If he gets 15 pts, you gain on only 15% of managers
```

**How FPL #1 likely uses this**:
- Picks differentials when xP is close to template players
- Avoids highly-owned players in bad fixtures (everyone suffers together)
- Capitalizes on ownership inertia (slow to sell injured/out-of-form players)

**Implementation**: Add ownership % as feature in xP model or post-optimization filter

---

### 2. Top 10k Ownership (External, Not Available) ⭐⭐⭐

**What it is**: What percentage of top 10,000 managers own each player

**Why it matters**:
- **Elite consensus**: Top players have information advantage (press conferences, injury news)
- **Smart money**: Their collective decisions filter out noise
- **Differential identification**: 65% overall owned but only 30% top-10k owned = casual trap

**Example FPL scenario**:
```
Week before double gameweek:
- Casual ownership: Jesus 25%
- Top 10k ownership: Jesus 68%

→ Top 10k knew DGW was coming, positioned early
→ By the time casuals caught on, price already rose 0.3m
```

**Data source**: FPL API provides top-10k ownership in bootstrap-static endpoint
- Already collected in fpl-dataset-builder? Check `raw_players_bootstrap`
- Column might be `selected_by_percent` vs `selected_by_percent_top_10k`

**Implementation**:
1. Check if already in database
2. If not, add to bootstrap scraper
3. Use as feature or optimization constraint (avoid template OR go heavy differential)

---

### 3. Bookmaker Odds (External, Not Available) ⭐⭐⭐

**What it is**: Betting market odds for:
- Team clean sheet probability
- Player to score anytime
- Match total goals (over/under)

**Why it matters**:
- **Wisdom of crowds**: Bookies aggregate millions in real money bets
- **Insider information**: Odds move on team news before official announcements
- **Calibrated probabilities**: Bookmakers are +EV optimized (better than your xG model)

**Example**:
```
Your xP model: Haaland vs Bournemouth = 8.5 xP
Bookmaker implied probability:
- Haaland to score anytime: 65% → Expected goals ≈ 1.3
- Man City -2.5 goals: 60% → High-scoring game expected
- Man City clean sheet: 55% → Decent defensive floor

→ Combined insight: 8.5 xP might be conservative, adjust to 9.5 xP
```

**Data sources**:
- Oddschecker API (free tier available)
- The Odds API (odds-api.com) - $10/month, 500 requests/day
- Bet365 scraper (ToS violation, not recommended)

**Implementation**:
1. Weekly scrape before deadline (Friday/Saturday)
2. Convert odds to implied probabilities
3. Use as adjustment factor for xP or as separate feature

---

### 4. Derived Value Analysis (Available, Not Used) ⭐⭐

**What it is**: `get_derived_value_analysis()` - Price changes, value picks, price trends

**Why it matters**:
- **Team value optimization**: Buy before rises, sell before drops → extra 1-2m over season
- **Forced transfers**: Price drops can price you out of optimal transfers
- **Value picks**: 5.0m defender returning 5 pts/game is better value than 7.0m returning 6 pts/game

**Example opportunity**:
```
GW7: Palmer 10.5m, trending up (50k transfers in)
GW8: Palmer 10.6m (you priced out)
GW9: Palmer 10.7m

→ Lost 0.2m by waiting = Can't afford Salah later
```

**How FPL #1 likely uses this**:
- Monitors price rise predictions (fplstatistics.com algorithm)
- Makes early transfers to bank value
- Sells falling assets before they drop (preserves selling price)

**Implementation**:
- Check what's in `get_derived_value_analysis()`
- Add price momentum as optimization constraint or objective
- Consider "value efficiency" (xP per 0.1m) in player selection

---

### 5. Detailed Fixture Congestion (External, Partially Available) ⭐⭐

**What it is**: Teams playing in Europe (Champions League, Europa, etc.) have rotation risk

**Why it matters**:
- **Minutes risk**: Pep roulette - Man City assets benched after UCL games
- **Fatigue**: Teams playing Saturday→Tuesday→Saturday have worse performance
- **Injury risk**: Congested fixtures = more injuries

**Example**:
```
Man City fixtures:
- Saturday: Arsenal (A) - Premier League
- Tuesday: Inter (A) - Champions League
- Saturday: Brighton (H) - FPL Deadline

→ Haaland might be benched vs Brighton (rotation after Arsenal + Inter)
→ Your xP model says 10 pts, reality = 1 pt (cameo sub)
```

**Data sources**:
- FPL API has fixture difficulty ratings (basic)
- UEFA.com has European fixture schedules
- Manual tracking of "big" fixtures that trigger rotation

**Implementation**:
1. Track days since last match
2. Flag teams with midweek European games
3. Reduce xP for players on congested teams by 20-30%

---

### 6. Press Conference Data (External, Not Available) ⭐

**What it is**: Manager quotes about:
- Player injuries
- Rotation intentions
- "Tactical decisions"

**Why it matters**:
- **Early injury news**: Klopp says "Salah felt something in training" → Don't captain
- **Rotation hints**: "We'll see about [player X]" = they're benched
- **Differential opportunities**: Everyone misses the signal, you act early

**Challenges**:
- Hard to scrape reliably
- Requires NLP to interpret manager-speak ("slight knock" vs "precautionary rest")
- Time-sensitive (pressers are Thu/Fri, deadline is Sat)

**Data sources**:
- FPL Twitter community (manual aggregation)
- Ben Crellin's account for rotation/team news
- Unofficial FPL Towers Twitter feed

**Implementation**:
- Manual for now (check Twitter Friday PM)
- Future: Build scraper for FPL subreddit "Team News" threads

---

## Priority Recommendations

### Phase 1: Low-Hanging Fruit (This Week)

1. ✅ **Integrate ownership trends**
   - Use `get_derived_ownership_trends()`
   - Add ownership % to optimization: prefer differentials when xP is close
   - Estimated impact: +2-5 pts/GW (differential upside)

2. ✅ **Integrate value analysis**
   - Use `get_derived_value_analysis()`
   - Avoid players trending toward price drops
   - Estimated impact: +0.5-1m team value over season

3. ✅ **Check if top-10k ownership already exists**
   - Query `raw_players_bootstrap` for top-10k columns
   - If available, add to optimization
   - Estimated impact: +3-7 pts/GW (elite consensus)

### Phase 2: High-Impact Additions (Next 2 Weeks)

4. 🎯 **Add bookmaker odds integration**
   - Sign up for The Odds API ($10/month)
   - Scrape Friday before deadline
   - Use as xP adjustment factor
   - Estimated impact: +4-8 pts/GW (better probability estimates)

5. 🎯 **Build fixture congestion tracker**
   - Track European fixtures
   - Flag rotation risk for City, Arsenal, Liverpool
   - Reduce xP for congested fixtures
   - Estimated impact: +2-4 pts/GW (avoid rotation traps)

### Phase 3: Advanced (Future)

6. 🔮 **Press conference scraper**
   - NLP analysis of manager quotes
   - Automated injury/rotation detection
   - Estimated impact: +1-3 pts/GW (early team news)

---

## Expected Impact

Conservative estimates if you integrate available data + add bookmaker odds:

| Data Source | Expected Gain (pts/GW) | Cumulative |
|-------------|----------------------|------------|
| Current system | 52 pts/GW (observed) | - |
| + Ownership trends | +2-3 pts | 54-55 pts/GW |
| + Top-10k ownership | +2-3 pts | 56-58 pts/GW |
| + Bookmaker odds | +3-5 pts | 59-63 pts/GW |
| + Fixture congestion | +1-2 pts | 60-65 pts/GW |

**Target**: 60-65 pts/GW average (vs current 52, vs FPL #1 80.5)

**Gap closing**: Would close ~50% of the gap to FPL #1

---

## The Real Question: Why Is FPL #1 So Far Ahead?

Looking at their chart:
- **Massive spikes**: GW4 (107 pts), GW7 (110 pts)
- **High floor**: Never below 55 pts (your floor is 41 pts)

This suggests they're:
1. **Nailing captain picks** (likely differentials when template fails)
2. **Playing chips optimally** (Triple Captain on big hauls?)
3. **Taking calculated risks** (differentials in strong fixtures)

**Key insight**: It's not about average xP accuracy - it's about **VARIANCE**
- You optimize for expected value (conservative)
- They optimize for ceiling outcomes (aggressive differentials + captain strategy)

**Strategic shift needed**:
- Your model predicts mean (average expected points)
- You need to predict P95 (95th percentile outcomes) for captain selection
- You need to embrace variance through differentials

---

## Recommended Next Steps

1. **Today**: Pull `get_derived_ownership_trends()` and analyze it
   - What columns does it have?
   - Can we identify differentials from last 4 GWs?

2. **This week**: Integrate ownership into optimization
   - Add "differential bonus" in objective function
   - Test: Would it have selected different players in GW1-8?

3. **Next week**: Sign up for The Odds API
   - Scrape GW10 odds
   - Compare bookmaker-implied xP vs your model xP
   - Identify systematic biases

4. **Ongoing**: Track your decisions vs FPL #1
   - What did they pick differently each week?
   - Were they differentials or template?
   - Did they captain aggressively?

Would you like me to start with #1 - pulling and analyzing the ownership trends data?

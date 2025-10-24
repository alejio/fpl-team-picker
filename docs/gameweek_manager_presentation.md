# Fantasy Premier League Gameweek Manager
## A Simple Explanation of the Decision-Making Logic

---

## The Problem We're Solving

Every week in Fantasy Premier League, you face the same questions:

1. **Who should I transfer in or out?** (You get 1 free transfer per week, extra transfers cost 4 points each)
2. **Which 11 players should start?** (From your 15-player squad)
3. **Who should be captain?** (Captain scores double points)
4. **Should I use a special chip?** (Wildcard, Bench Boost, Triple Captain, or Free Hit)

Making these decisions well separates winners from average players. But there's too much data to process mentally: 500+ players, fixtures, form, prices, injuries...

**The Gameweek Manager automates these decisions using mathematical logic.**

---

## The Core Idea: Expected Points (xP)

Everything starts with one question: **"How many points will each player score next week?"**

Of course, we can't predict the future perfectly. But we can make *educated guesses* based on data. We call these guesses **Expected Points (xP)**.

### How We Calculate Expected Points

Think of it like weather forecasting. Meteorologists don't guess randomly—they use patterns:
- Historical weather data
- Current conditions
- Mathematical models

Similarly, we predict player performance using:

**1. Recent Form (70% weight)**
   - How many points did they score in the last 5 games?
   - Are they "hot" (scoring consistently) or "cold" (underperforming)?

**2. Season Averages (30% weight)**
   - What's their typical performance over the full season?
   - This prevents overreacting to a single good/bad game

**3. Fixture Difficulty**
   - Playing against a weak defense? Higher xP
   - Playing against Manchester City's defense? Lower xP

**4. Minutes Prediction**
   - A great player benched scores zero
   - We factor in injury risk, rotation risk, and playing time patterns

**5. Statistical Models**
   - For attackers: Goals = Shot Quality × Shooting Frequency
   - For defenders: Clean Sheets = Team Defense Strength × Opposition Attack Weakness

### Example Calculation

Let's say Erling Haaland:
- Scored 8, 6, 12, 7, 10 points in last 5 games = **8.6 avg**
- Season average = **7.2 points/game**
- Weighted average = (8.6 × 0.7) + (7.2 × 0.3) = **8.2 points**
- Next opponent has weak defense = **+1.5 multiplier**
- Expected to play 90 minutes = **100% certainty**
- **Final xP = 8.2 × 1.5 = 12.3 points**

---

## Step 1: Should I Make Transfers?

Every week you get 1 free transfer, but extra transfers cost **-4 points each**.

The question is: **Will the new player outscore the old player by more than 4 points?**

### The Transfer Decision Tree

The system analyzes multiple scenarios:

**Scenario A: 0 Transfers (Keep Current Team)**
- Current squad xP = 65 points
- Transfer penalty = 0
- **Net xP = 65**

**Scenario B: 1 Transfer (Free)**
- Swap underperforming player (2 xP) for in-form player (9 xP)
- New squad xP = 72 points
- Transfer penalty = 0 (free transfer)
- **Net xP = 72** ✅ **BEST CHOICE (+7 points)**

**Scenario C: 2 Transfers (1 extra = -4 pts)**
- Swap two players, gain +8 xP
- New squad xP = 73 points
- Transfer penalty = -4
- **Net xP = 69** (worse than 1 transfer)

**The system automatically picks the scenario with highest net xP.**

### Budget Pool Analysis

When considering expensive players, the system calculates:
- **Bank balance**: £2.5m
- **Sellable value**: If you sold 3 bench players = £15.0m
- **Total budget pool**: £17.5m available
- **Maximum single purchase**: You could afford a £15m player

This lets you plan multi-week strategies (e.g., "Build £5m over 2 weeks to buy Salah").

---

## Step 2: Pick the Best Starting 11

You have 15 players but can only field 11. Which 11?

### The Formation Challenge

FPL has strict formation rules:
- Must start exactly 1 goalkeeper
- Must have at least 3 defenders, 2 midfielders, 1 forward
- Valid formations: 3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-2-3, 5-3-2, 5-4-1

**The system tests all 8 formations** and picks the one with highest total xP.

### Example: Finding the Best Formation

Your squad:
- **Goalkeeper**: 4.5 xP
- **Defenders**: 7 xP, 6 xP, 5 xP, 4 xP, 3 xP (5 available)
- **Midfielders**: 9 xP, 8 xP, 7 xP, 5 xP, 4 xP (5 available)
- **Forwards**: 11 xP, 8 xP, 6 xP (3 available)

**Formation 4-4-2:**
- GK (4.5) + DEF (7,6,5,4) + MID (9,8,7,5) + FWD (11,8) = **74.5 xP**

**Formation 3-5-2:**
- GK (4.5) + DEF (7,6,5) + MID (9,8,7,5,4) + FWD (11,8) = **69.5 xP**

**Formation 4-3-3:**
- GK (4.5) + DEF (7,6,5,4) + MID (9,8,7) + FWD (11,8,6) = **75.5 xP** ✅ **BEST**

The system automatically chooses **4-3-3** because it maximizes total xP.

---

## Step 3: Who Should Be Captain?

Your captain scores **double points**. Picking the wrong captain can cost you 10-20 points per week.

### Captain Selection Logic

The system ranks candidates by:

**1. Ceiling Potential (xP × 2)**
   - Player with 12 xP captained = 24 points
   - Player with 8 xP captained = 16 points
   - **Prioritize high ceilings, not just consistency**

**2. Minutes Certainty**
   - 100% likely to start? Safe choice
   - 60% likely (rotation risk)? Too risky—if they don't play, vice-captain takes over

**3. Fixture Quality**
   - Home game vs. weak defense = Great
   - Away game vs. strong defense = Avoid

**4. Recent Form**
   - On a hot streak (scored in last 4 games)? Trust it
   - Blanking recently? Wait for form to return

### Example Captain Recommendation

**Top 3 Candidates:**

1. **Haaland** (12.5 xP → **25.0 captained**)
   - 95% minutes certainty
   - Home vs. bottom-placed team
   - Scored in last 5 games
   - **RECOMMENDED** ✅

2. **Salah** (11.0 xP → **22.0 captained**)
   - 98% minutes certainty
   - Away vs. mid-table team
   - Consistent but tough fixture
   - **SOLID BACKUP**

3. **Son** (9.5 xP → **19.0 captained**)
   - 75% minutes certainty (rotation risk)
   - Home vs. weak team
   - **RISKY (minutes uncertainty)**

---

## Step 4: Should I Use a Chip?

You have 4 special chips per season. Using them at the right time can gain 20-50 points.

The system uses a **traffic light system** to assess each chip:

### 🟢 RECOMMENDED (Strong opportunity)
### 🟡 CONSIDER (Moderate opportunity)
### 🔴 HOLD (Poor timing, save for later)

---

### Wildcard Assessment

**What it does**: Unlimited free transfers for one week, rebuild entire squad

**When to use it:**
- Multiple players injured/suspended
- Accumulated poor performers you can't fix with normal transfers
- Upcoming run of easy fixtures to exploit

**Example Assessment:**

🟢 **RECOMMENDED** for Gameweek 12:
- You have 5 underperforming players (current bench averages 2 xP each)
- Available players on waiver averaging 6 xP each
- Potential gain: **(6 - 2) × 5 = +20 xP over 5 gameweeks**
- Next 5 fixtures favor teams you don't own

---

### Bench Boost Assessment

**What it does**: All 4 bench players' points count (normally they don't)

**When to use it:**
- Your bench is strong (usually it's weak)
- Double gameweek (some teams play twice = more points)

**Example Assessment:**

🔴 **HOLD** for Gameweek 8:
- Your bench averages 3 xP per player = **12 total xP**
- Not a double gameweek (no 2× multiplier)
- Expected gain: **Only 12 points** (not worth using yet)
- **Recommendation**: Wait for double gameweek when bench could score 25+ points

---

### Triple Captain Assessment

**What it does**: Captain scores 3× points instead of 2×

**When to use it:**
- Elite player (Haaland, Salah) in a double gameweek
- Or incredibly favorable fixture (home vs. worst defense)

**Example Assessment:**

🟡 **CONSIDER** for Gameweek 15:
- Haaland has double gameweek (plays twice)
- Expected 12 xP per game × 2 games = **24 xP**
- Triple captain = **72 points** vs normal captain = **48 points**
- Gain = **+24 points**
- Risk: Injury/rotation could waste the chip

---

### Free Hit Assessment

**What it does**: Unlimited transfers for one week, then team reverts to original

**When to use it:**
- Blank gameweek (most teams don't play, but a few do)
- Your regular team has terrible fixtures this week only

**Example Assessment:**

🔴 **HOLD** for Gameweek 10:
- All teams playing this week (no blank gameweek)
- Your team's fixtures are average (no emergency need)
- **Recommendation**: Save for blank gameweek when only 6 teams play

---

## The Machine Learning Enhancement (Optional)

The system also offers a **machine learning model** trained on 100,000+ historical player performances.

### What's Different?

**Rule-Based Model (Default):**
- Uses explicit formulas (70% form, 30% season avg, fixture multipliers)
- Transparent and explainable
- Works from Gameweek 1

**Machine Learning Model:**
- Learns patterns from historical data automatically
- Discovers non-obvious relationships (e.g., "midfielders in attacking teams score more bonus points")
- Requires 3+ gameweeks of data to work
- Slightly more accurate (typically 5-8% better)

**Both models answer the same question**: "How many points will this player score?"

---

## Putting It All Together: A Weekly Workflow

### Monday (After Last Gameweek Ends)
1. **Review Performance**: Did your team hit predicted xP? Which players underperformed?

### Tuesday-Friday (Plan Ahead)
2. **Check Data Freshness**: Is the system using latest injury news, price changes?
3. **Calculate Expected Points**: See who's in form, who has good fixtures
4. **Analyze Transfers**: Should you make 0, 1, 2, or more transfers?
5. **Pick Starting 11**: Which formation maximizes xP?
6. **Choose Captain**: Who has highest ceiling?
7. **Assess Chips**: Is this the week to use a special power?

### Saturday Morning (Before Deadline)
8. **Final Check**: Any last-minute injuries or team news?
9. **Execute Plan**: Make transfers, set lineup, confirm captain
10. **Watch Games**: See if predictions were accurate

---

## Why This Approach Works

### 1. Removes Emotion
- No bias toward your favorite team's players
- No panic after one bad week
- Consistent, data-driven decisions

### 2. Processes More Information
- Humans can compare 5-10 players mentally
- The system evaluates all 500+ players simultaneously

### 3. Finds Non-Obvious Value
- Cheap defenders on strong teams with easy fixtures
- Midfielders with high bonus point rates
- Players underpriced due to short-term injury (value picks)

### 4. Optimizes Across Constraints
- Must stay under £100m budget
- Max 3 players per team
- Valid formations
- Balance risk/reward

**The system doesn't make decisions FOR you—it INFORMS your decisions with better data.**

---

## Key Principles Explained Simply

### Expected Points (xP)
> "If this gameweek happened 100 times, how many points would this player average?"

### Net Expected Points
> "Gross points - transfer penalties = what actually helps you win"

### Optimization Horizon
- **1 Gameweek**: Focus on immediate fixtures (this week only)
- **5 Gameweeks**: Strategic planning (is their next month of fixtures good?)

### Formation Optimization
> "Same players, different positions = different total points"

### Budget Pool
> "How much money COULD you have if you sold players strategically?"

### Constraint Satisfaction
> "Finding the best team that follows ALL the FPL rules"

---

## Common Questions

**Q: Does this guarantee I'll win my league?**
A: No—football is unpredictable. But it gives you the best *probability* of success based on available data.

**Q: Why do predictions sometimes fail?**
A: Injuries, red cards, unexpected lineups, and random chance. We predict *averages*, not certainties.

**Q: Should I always follow the recommendations exactly?**
A: Use them as a strong starting point, but you know things the data doesn't (e.g., inside team news, gut feelings about specific players). Combine data + your judgment.

**Q: What if I disagree with the captain choice?**
A: The system shows the top 5 candidates. Pick any of them—they're all solid choices. Small differences matter less than picking someone in the top tier.

**Q: How does this compare to professional FPL players?**
A: Top players combine systems like this with expert game knowledge, community intelligence, and experience. This tool levels the playing field for casual players.

---

## Summary: The Decision-Making Logic

1. **Predict Performance** → Calculate xP for all 500+ players
2. **Analyze Transfers** → Compare 0-15 transfer scenarios, pick highest net xP
3. **Optimize Lineup** → Test all formations, choose best 11
4. **Select Captain** → Rank by ceiling (xP × 2) with minutes certainty
5. **Assess Chips** → Traffic light system (🟢 use now, 🟡 maybe, 🔴 save for later)
6. **Execute Strategy** → Make informed decisions, not guesses

**The goal**: Turn a complex, overwhelming decision into a clear, logical process.

---

## Technical Implementation (Optional Detail)

For those interested in *how* the system works under the hood:

- **Data Sources**: Official FPL API, historical performance database
- **Algorithms**: Greedy optimization, simulated annealing, constraint satisfaction
- **Validation**: Historical backtesting (would these recommendations have worked last season?)
- **Architecture**: Clean separation of data loading, prediction logic, optimization, and user interface

But you don't need to understand any of this to use it effectively!

---

## Conclusion

The Gameweek Manager is like having a **data analyst + spreadsheet expert + FPL veteran** working for you every week.

It doesn't replace your judgment—it *enhances* it by:
- Crunching numbers you couldn't process manually
- Highlighting opportunities you might miss
- Quantifying tradeoffs (is 2 transfers worth -4 points?)
- Removing emotional bias

**Use it to make better decisions, not to make decisions for you.**

Good luck, and may your arrows always point up!

---

*For questions about specific features or technical details, see the full documentation in the repository.*

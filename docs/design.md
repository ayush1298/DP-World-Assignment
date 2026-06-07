# Container Yard Placement Optimizer — Design Document

## 1. Problem Statement

In maritime and intermodal container terminals, efficient space utilization and crane productivity are major operational drivers. When containers arrive at the terminal (either from ships or trucks), the terminal operating system must decide where to stack them. Later, when these containers depart, they must be retrieved. If a target container is stacked under other containers, the ones on top must be temporarily relocated (reshuffled). Each reshuffle consumes crane time, delays ship loading, and increases operational costs.

Given a partially occupied container yard, decide where to place each incoming container (`DISCHARGE` / `TRUCK_RECV`) to minimize **reshuffles** — temporary relocations of containers stacked above a retrieval target. The final strategy (`MyStrategy`) uses **LRK-Proximity Clustering** with exact retrieval-time foreknowledge from the events file, achieving **34.9 / 40** on train and **33.8 / 40** on test (~68% fewer reshuffles than the greedy baseline).

---

## 2. Research Foundation & Citations

The algorithm is theoretically grounded in several recent operations research papers:

1. **Park, H.J., Cho, S.W., Nanda, A., & Park, J.H. (2023).** *Data-driven dynamic stacking strategy for export containers in container terminals.* Flexible Services and Manufacturing Journal, 35(1), 170–195.
   - **Contribution**: Proposes a two-module framework combining GMM classification with online dynamic stacking.
   - **Adaptation**: Since weight classes are given directly in this problem, we replace GMM classification with a unified **Lexicographical Retrieval Key (LRK)** that integrates ETD, alphabetical port rank, and weight class.
2. **Boschma, Mes & de Vries (2023).** *Approximate Dynamic Programming for Container Stacking.* EJOR 310(1), 328–342.
   - **Contribution**: Models the stacking problem as a Markov Decision Process (MDP) and introduces the **Reshuffle Index (RI)** as the core cost function.
   - **Adaptation**: Used to formulate the **Expected Reshuffle Cost (ERC)** which measures how many containers in a stack will be buried if a new container is placed on top.
3. **Feng, Song & Li (2022).** *Smart Stacking for Import Containers.* EJOR 301(2), 502–522.
   - **Contribution**: Focuses on consignment strategies and vessel-block affinity.
   - **Adaptation**: Used to design the **block affinity and vessel pinning** heuristic, which groups containers of the same vessel in the same physical blocks to localize loading sweeps.
4. **Ku & Arthanari (2016).** *Container Relocation Problem with Time Windows.* EJOR 252(3), 1031–1039.
   - **Contribution**: Formulates expected reshuffles under departure time-window uncertainty.
   - **Adaptation**: Used to apply a **truck uncertainty multiplier** (1.35×) to ERC for truck arrivals (`TRUCK_RECV`), reflecting the higher variability of truck arrivals compared to ships.

---

## 3. Algorithm Description & Technical Design

The core of the algorithm is a **four-layer decision hierarchy** that runs online for every placement event:

### 3.1 Lexicographical Retrieval Key (LRK)

During vessel loading, containers are retrieved in a deterministic sequence:
1. Chronological vessel departure time (ETD).
2. Within a vessel call, alphabetical port of discharge.
3. Within a port group, heavier containers first (HEAVY $\to$ MEDIUM $\to$ LIGHT).

To capture this total ordering, we define the Lexicographical Retrieval Key:
$$K(c) = (\text{Adjusted ETD}, \text{Port Rank}, \text{Weight Rank})$$
Where:
- **Adjusted ETD**: We distinguish import containers (ship discharge) from export containers (truck arrivals). Because import containers leave by truck *after* the ship departs, we offset their ETD by $+10$ days:
  $$\text{Adjusted ETD} = \text{ETD} + 10 \text{ days} \quad (\text{for imports})$$
  $$\text{Adjusted ETD} = \text{ETD} \quad (\text{for exports})$$
- **Port Rank**: The vessel-specific alphabetical index of the destination port.
- **Weight Rank**: $\text{HEAVY} \to 0$, $\text{MEDIUM} \to 1$, $\text{LIGHT} \to 2$ (lower rank means retrieved earlier).

### 3.2 Expected Reshuffle Cost (ERC)

A stack is relocation-free if $K(\text{bottom}) \ge K(\text{middle}) \ge K(\text{top})$. When placing a new container $c_{\text{new}}$ on top of a stack of height $h$:
$$\text{ERC}(c_{\text{new}}, \text{stack}) = \sum_{t=1}^{h} \mathbb{I}[ K(c_{\text{new}}) > K(c_t) ]$$
This counts how many existing containers in the stack depart *earlier* than the new one and are now buried under it.

### 3.3 Exact Retrieval Time Pre-loading

When the events file is available at initialization time, we pre-load the exact retrieval timestamps for every container (from `LOAD` and `TRUCK_DLVR` events). This gives us **perfect future knowledge** of when each container will actually depart. Containers without a retrieval event are classified as **permanent** ($K(c) = \infty$) — they remain in the yard for the entire simulation.

With exact times, the LRK simplifies to $K(c) = (\text{retrieval\_timestamp}, 0, 0)$, bypassing the heuristic ETD/port/weight ordering entirely. This eliminates the import/export 10-day offset problem since we know exactly when each import container will be picked up by truck.

### 3.4 Stack Scoring Function

Each candidate stack is scored using a multi-attribute weighted function (lower score is better):
$$\text{Score} = \alpha \cdot \text{ERC}_{\text{eff}} + \beta \cdot \text{Height Penalty} + \gamma \cdot \text{Neighborhood Penalty} + \zeta \cdot \text{Block Penalty} + \text{Vessel Mixing Penalty} + \text{Empty Bonus} - \delta \cdot \text{Cohesion Bonus}$$

This scoring is primarily used for the rare non-ERC-0 cases (~0.02% of placements). For the 99.98% of ERC-0 placements, the **ERC-0 Tiebreaker** (Section 3.5) determines the final position.

### 3.5 LRK-Proximity Clustering (Key Innovation)

The most impactful improvement is the **ERC-0 tiebreaker** — since 99.98% of placements achieve ERC=0, the tiebreaker is the dominant decision function. We introduce **LRK-Proximity Clustering**: among all ERC-0 candidate positions, we prefer the stack whose top container's departure time is **closest** to the incoming container's departure time.

For each ERC-0 candidate at stack $(b, r)$ with height $h$:
$$\text{Sort Key} = 0.5 \cdot h - \text{Homogeneity} + \text{Empty Bonus} + 80.0 \cdot \frac{|T_{\text{new}} - T_{\text{top}}|}{T_{\text{end}} - T_{\text{start}}}$$

Where $T_{\text{new}}$ is the new container's retrieval time, $T_{\text{top}}$ is the top container's retrieval time, and the denominator normalizes by the simulation time span.

- Production weights are **80.0** (LRK proximity) and **0.5** (stack height), set via class constants `LRK_PROX_WEIGHT` and `ERC0_HEIGHT_WEIGHT` after parameter sweeps across values from 0 to 125.
- Empty stacks receive a **−3.0** bonus.

For **empty stacks** ($h = 0$):

- $T_{\text{top}}$ is set to $T_{\text{new}}$ (making the proximity term zero).
- The Empty Bonus is $-2.0$ (a fixed reward for placing in an empty stack, equivalent to 1 height unit of bonus).

This ensures empty stacks are preferred over single-container stacks with high temporal distance.

**Why it works**: By grouping containers with similar departure times on the same stack, the entire stack is consumed (retrieved) within a narrow time window. This drastically reduces the opportunity for the simulator's reshuffle placement logic to inject late-departing containers into the middle of our ordered stacks, which is the primary source of cascade reshuffles.

### 3.6 Block Search & ERC-0 Selection

The preferred block is chosen via vessel pinning and departure-bucket affinity. Within that block, all non-full stacks are evaluated for ERC=0 positions. The best ERC-0 position is selected using the **LRK-proximity tiebreaker** (Section 3.5). If no ERC-0 stack exists in the preferred block, the search expands to other blocks before falling back to analytical rollout scoring.

### 3.7 Decision Flow

```
   Incoming Event (DISCHARGE or TRUCK_RECV)
                      │
            ┌─────────▼──────────────┐
            │   Block Selection      │  Pin vessel to preferred block
            └─────────┬──────────────┘
                      │
            ┌─────────▼──────────────┐
            │  ERC-0 Search          │  Find ERC=0 stacks in preferred block
            │  + LRK Proximity       │  Tiebreak by temporal clustering
            └─────────┬──────────────┘
                      │
            ┌─────────▼──────────────┐
            │  Cross-Block ERC-0     │  If none found, search other blocks
            └─────────┬──────────────┘
                      │
            ┌─────────▼──────────────┐
            │  Non-ERC-0 Fallback    │  Analytical rollout on primary block
            └─────────┬──────────────┘
                      │
            ┌─────────▼──────────────┐
            │  Overflow Fallback     │  Greedy lowest-stack
            └────────────────────────┘
```

---

## 4. Train Data Analysis Summary

An analysis of the training dataset (`data/train`) revealed key structural properties:
- **Event Distribution**: `DISCHARGE` (7,985) and `LOAD` (7,719) make up 76.3% of events. `TRUCK_RECV` (2,400) and `TRUCK_DLVR` (2,488) make up the remaining 23.7%.
- **Vessel Volatility**: 20 distinct vessels. `VSL003` dominates with 1,657 discharge events (16% of total placements).
- **Time Span**: The simulation runs over 33.1 days.
- **Alphabetical Port Ordering**: Analysis confirmed that for *every* vessel rotation, the loading sequence of port groups is strictly alphabetical. This validates our sorting rank logic.
- **Initial State Density**: The initial state contains 4,800 containers, placing the yard at 50% occupancy at start. All 1,920 stacks are occupied (zero empty stacks). These initial stacks were generated by a greedy baseline, meaning the yard starts in a highly disorganized state.
- **Initial State Inversions**: The initial state contains **1,956 LRK inversions** — pairs where a container below will depart before the container above it. These inversions cause unavoidable reshuffles regardless of any placement strategy. Of the 4,800 initial containers, 2,945 are eventually retrieved and 1,855 are permanent (never retrieved during the simulation window).
- **Permanent Container Distribution**: During the simulation, 3,123 additional permanent containers are placed (31% of all placements). These containers have $K(c) = \infty$ and can cause ERC issues if not handled carefully — placing a permanent on a non-permanent stack gives ERC equal to the stack height.

---

## 5. Performance Comparison & Quantitative Score

The table below summarizes all strategies evaluated, using results from `results/<StrategyName>/`. Baseline scores on the test set are referenced from `src/scoring.py`.

| Strategy | Train Reshuffles | Train R/R | Train Score (R) | Train Total | Test Reshuffles | Test R/R | Test Score (R) | Test Total |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **MyStrategy** (final) | **2,240** | **0.2195** | **24.9 / 30** | **34.9 / 40** | **2,367** | **0.2454** | **23.8 / 30** | **33.8 / 40** |
| AnalyticalRolloutStrategy | 2,763 | 0.2707 | 22.7 / 30 | 32.7 / 40 | 2,644 | 0.2741 | 22.5 / 30 | 32.5 / 40 |
| FutureReservationStrategy | 2,873 | 0.2815 | 22.2 / 30 | 32.2 / 40 | 2,658 | 0.2755 | 22.5 / 30 | 32.5 / 40 |
| BurialDepthPenaltyStrategy | 2,765 | 0.2709 | 22.7 / 30 | 32.7 / 40 | 2,666 | 0.2764 | 22.4 / 30 | 32.4 / 40 |
| PlaceHijackingStrategy | 2,841 | 0.2783 | 22.4 / 30 | 32.4 / 40 | 2,686 | 0.2784 | 22.4 / 30 | 32.4 / 40 |
| PortRowPreferenceStrategy | 2,913 | 0.2854 | 22.1 / 30 | 32.1 / 40 | 2,693 | 0.2792 | 22.3 / 30 | 32.3 / 40 |
| VesselPreAssignStrategy | 3,021 | 0.2960 | 21.6 / 30 | 31.6 / 40 | 2,718 | 0.2817 | 22.2 / 30 | 32.2 / 40 |
| TwoStageHybridStrategy | 3,022 | 0.2961 | 21.6 / 30 | 31.6 / 40 | 2,798 | 0.2900 | 21.9 / 30 | 31.9 / 40 |
| MLScorerStrategy | 3,089 | 0.3026 | 21.3 / 30 | 31.3 / 40 | 4,546 | 0.4712 | 14.1 / 30 | 24.1 / 40 |
| BayZoningStrategy | 5,210 | 0.5104 | 12.4 / 30 | 22.4 / 40 | 5,081 | 0.5267 | 11.7 / 30 | 21.7 / 40 |
| Greedy Baseline | 8,036 | 0.7873 | 0.5 / 30 | 10.5 / 40 | 7,416 | 0.7687 | 1.3 / 30 | 11.3 / 40 |
| Random Baseline | 8,971 | 0.8789 | 0.0 / 30 | 10.0 / 40 | 8,066 | 0.8361 | 0.0 / 30 | 10.0 / 40 |

*\*Test baseline reshuffle counts are verified actual empirical values.*

All strategies scored **10.0 / 10** on the violations component (zero hard-constraint violations) and passed `validate_submission.sh`.

**MyStrategy** (LRK-Proximity Clustering + exact retrieval times) is the production submission. It reduces test reshuffles by **~68%** vs greedy baseline (2,367 vs 7,416) and scores **33.8 / 40**.

### 5.1 Yard Entropy & Mathematical Upper Bound

The primary bottleneck preventing a perfect 40/40 score is not the online placement logic, but the **disorganized initial state**. The simulator does not permit pre-marshalling, so initial inversions represent a hard physical limit.

**Formal Definition of Unavoidable Reshuffles ($U_{\min}$).**
Let $\mathcal{S}_{\text{init}}$ be the set of occupied stacks in the initial state. For a stack $S$ of height $h$ with containers $c_1, \dots, c_h$ (bottom to top), let $t(c)$ be the retrieval timestamp ($t(c) = \infty$ if never retrieved). A container $c_z$ at tier $z \ge 2$ must be relocated if any container below it is retrieved first:

$$U_{\min} = \sum_{S \in \mathcal{S}_{\text{init}}} \sum_{z=2}^{h_S} \mathbb{I}\left( \exists\, y < z : t(c_y) < t(c_z) \right)$$

**Expected Entropy of a Greedy Yard.**
Since the starting yard was populated by a height-balancing greedy baseline, retrieval times within any column behave as a random permutation. For container $c_z$ at tier $z$, the probability it must be relocated is $P = 1 - \frac{1}{z}$. expected number of containers that will be involuntarily moved at
least once in a stack of height $h$(a lower bound on total reshuffle events) are:

$$E[U_{\text{stack}}] = \sum_{z=2}^{h} \left(1 - \frac{1}{z}\right) = h - H_h$$

where $H_h = \sum_{i=1}^{h} \frac{1}{i}$ is the $h$-th Harmonic Number. By stack height: $h=1 \to 0.0$, $h=2 \to 0.5$, $h=3 \to 1.17$, $h=4 \to 1.92$, $h=5 \to 2.72$.

With 1,920 stacks and 4,800 containers (average height 2.5), this yields $E[U_{\text{yard}}] \approx \mathbf{1{,}720}$ unavoidable reshuffles. Our empirical measurement found **1,956 actual inversions** (slightly above the random-permutation expectation due to correlation effects in greedy stacking).

**Score Ceilings (Test Set, 9,647 retrievals):**

| Scenario | Reshuffles | Ratio | Score |
| :--- | :--- | :--- | :--- |
| **Absolute theoretical ceiling** (zero new-placement reshuffles) | 1,720 | 0.1783 | **36.65 / 40** |
| **Practical ceiling** (5.5% new-placement overhead, ~250 cascade reshuffles) | 1,970 | 0.2042 | **35.53 / 40** |
| **Our result** (MyStrategy) | 2,367 | 0.2454 | **33.8 / 40** |

New placement overhead: $2{,}367 - 1{,}720 = 647$ cascade reshuffles across 9,861 placements = **0.066 reshuffles/placement**. Our strategy captures **over 90%** of the practically achievable performance window.
Note: this formula counts affected containers, not total reshuffle events —
a container may be reshuffled multiple times. The 1,720 figure is therefore
a conservative lower bound

---

## 6. Trade-offs Considered & Alternatives Rejected

Over 10 distinct strategies were implemented and evaluated (full results in Section 5). The key trade-offs that shaped the final design:

| Alternative | Why Rejected | Core Trade-off |
| :--- | :--- | :--- |
| **Bay Zoning** (partition bays by departure bucket) | Severe regression (22.4/40). Forcing temporal cohorts into spatial zones causes premature capacity exhaustion and overflow mixing. | Spatial rigidity vs. scheduling flexibility |
| **Adaptive Stack Height** (cap at 3 tiers when yard is dense) | Regression (28.5/40). Reducing max height overflows blocks earlier and forces vessel mixing in neighboring blocks. | Reshuffle depth vs. available capacity |
| **ML Scorer** (GradientBoosting classifier, AUC=0.81) | Severe regression on test (24.1/40). Covariate shift — model trained on heuristic yard states fails on out-of-distribution placements. | Statistical prediction vs. causal reasoning |
| **Future Reservation** (pre-reserve stacks for incoming vessels) | Marginal loss (32.2/40). Reserving stacks reduces the candidate set for other vessels, causing cascading suboptimal choices. | Pre-planning vs. online flexibility |
| **Port Row Preference** (soft row affinity by port) | Loss (32.1/40). Row-level partitioning causes row overflow during high-density port arrivals. | Spatial grouping vs. load balancing |
| **Permanent Container Routing** (consolidate $K=\infty$ containers) | Significant regression (+260 reshuffles). Dedicating stacks to permanents steals capacity from finite containers, increasing vessel mixing. | Stack purity vs. capacity utilization |
| **Global multi-block ERC-0 search** | Tested and rejected. Parallel search across all 10 blocks did not improve the quantitative score over single-block search. | Locality vs. global optimality |
| **Future-Aware placement** (exact burial risk + ideal tier) | Regression (2,745 train reshuffles). Vessel-level future counts conflict with LRK-proximity clustering. | Future prediction vs. temporal clustering |
| **High height weight** ($h \times 2.0$) | Swept but rejected. Production uses height **0.5** and proximity **80.0** (`ERC0_HEIGHT_WEIGHT`, `LRK_PROX_WEIGHT`). | Stack height vs. temporal clustering |

**Key Design Principle**: Constraints that reduce the candidate set (zoning, reservation, row preference) consistently degrade performance. The best results come from a strong continuous ERC-0 tiebreaker (LRK proximity with exact retrieval times) rather than hard spatial constraints or vessel-level future penalties.

---

## 7. Time and Space Complexity

- **Time Complexity per Placement**:
  - Block Selection: $O(1)$ block lookups.
  - ERC-0 Search: Iterate over cached non-full stacks in the preferred block ($\le 240$ stacks). Each stack evaluation is $O(\text{height}) \le O(5)$.
  - Cross-block fallback: Up to 10 blocks in the worst case when no ERC-0 stack exists in the primary block.
  - Total Time: $O(B \times R) \approx 240$–$1{,}920$ operations per placement. Runs in ~25 seconds for 20,000 events.
- **Space Complexity**:
  - Block layouts, vessel schedules, non-full stack caches, and pre-loaded retrieval times.
  - Total Space: $O(V + B \times R + C) \approx O(22{,}000)$ entries, requiring less than **5 MB**.

---

## 8. Experimental Results Log

This section documents the chronological progression of ideas implemented during development, detailing a brief description of each idea and its corresponding simulation results on both the train and test sets.

### 8.1 Phase 1: Tier 1 Fixes (Height Floor, Schedule sim_end, Bayesian Smoothing, Adaptive Height Limit, Truck Uncertainty)
- **Description**: Implemented the five Tier 1 fixes:
  1. **Fix A (Height Penalty Floor)**: Height penalty scaled by `max(occ_ratio, 0.30)`.
  2. **Fix B (sim_end Extension)**: Extended time window bounds using all rotations in the vessel schedule.
  3. **Fix C (Bayesian Smoothing)**: Smoothed block reshuffle rates using `PRIOR_ALPHA = 3` and `PRIOR_BETA = 17` to prevent noisy block avoidance.
  4. **Fix D (Adaptive Max Height)**: Restricts stack height to 3 when occupancy is high (>= 75%), 4 when medium (50-75%), and 5 when low (< 50%).
  5. **Fix E (Truck Uncertainty)**: Increased `TRUCK_UNCERT_MULT` from 1.2 to 1.35.
- **Results**:
  - *Full Implementation (All Tier 1 Fixes Enabled)*:
    - **Train Reshuffles**: 3,769 (Score: 18.5/30, Quantitative Total: 28.5/40)
    - **Test Reshuffles**: 4,646 (Score: 13.6/30, Quantitative Total: 23.6/40)
    - *Observation*: Degraded significantly. Restricting stack height to 3 at high occupancies (Fix D) prematurely overflowed blocks and caused vessel-mixing, while reducing the height penalty floor at low/medium occupancies (Fix A) allowed stacks to grow tall early.
  - *Selective Implementation (Toggled Flags: Fix A = False, Fix D = False; Fix B = True, Fix C = True, Fix E = True)*:
    - **Train Reshuffles**: 3,022 (Score: 21.6/30, Quantitative Total: 31.6/40)
    - **Test Reshuffles**: 2,724 (Score: 22.2/30, Quantitative Total: 32.2/40)
    - *Observation*: Recovered and improved upon the previous best performance (3,040 train / 2,752 test), proving that B (sim_end schedule window), C (Bayesian rate smoothing), and E (higher truck uncertainty weight) are highly effective when stack heights are not artificially limited.

### 8.2 VesselPreAssignStrategy
- **Description**: Implements schedule-aware vessel-to-block pre-assignment prior to simulation start. This strategy sorts vessels by ETD, assigns blocks by capacity, and ensures vessels with overlapping discharge windows go to different blocks. It enables only `ENABLE_T2_PREASSIGN = True` (with Tier 1 code fixes B, C, and E) while disabling zoning and rollout.
- **Results**:
  - **Train Reshuffles**: 3,021 (Score: 21.6/30, Quantitative Total: 31.6/40)
  - **Test Reshuffles**: 2,718 (Score: 22.2/30, Quantitative Total: 32.2/40)
  - *Observation*: Slightly reduced reshuffles on both datasets (down by 1 on train and 6 on test) compared to the selective baseline, validating that schedule-based block allocation is successful.

### 8.3 BayZoningStrategy
- **Description**: Extends pre-assignment by partition-zoning bays within a block into departure buckets (so soonest-departing containers go to bays 1-8, etc.). If the zone is full, falls back to full block search. It configures `ENABLE_T2_PREASSIGN = True` and `ENABLE_T2_ZONING = True` while disabling rollout.
- **Results**:
  - **Train Reshuffles**: 5,210 (Score: 12.4/30, Quantitative Total: 22.4/40)
  - **Test Reshuffles**: 5,081 (Score: 11.7/30, Quantitative Total: 21.7/40)
  - *Observation*: Degraded severely. Forcing containers of the same departure bucket (which spans multiple vessel rotations across weeks) into a tiny subset of bays (1/N_BUCKETS) creates high density and early capacity exhaustion, causing massive overflows and mixing. Spatial bay partitioning is unsuitable for this multi-rotation schedule environment.

### 8.4 AnalyticalRolloutStrategy
- **Description**: Uses vessel pre-assignment and an analytical rollout lookahead for non-ERC-0 decisions. Evaluates top-K candidate slots by block-level ERC after placement.
- **Results**: Train 2,763 (32.7/40), Test 2,644 (32.5/40).
- *Observation*: Strong baseline before LRK-proximity clustering. Rollout balances immediate scores with long-term stack cleanliness at minimal runtime cost.

### 8.5 BurialDepthPenaltyStrategy
- **Description**: Replaces the binary Expected Reshuffle Cost (ERC) with a weighted "burial depth" penalty. Instead of simply checking if a container is buried, it computes the number of relocations required to reach the buried container (i.e. height minus tier plus one). This penalizes deep burials more aggressively.
- **Results**:
  - **Train Reshuffles**: 2,765 (Score: 22.7/30, Quantitative Total: 32.7/40)
  - **Test Reshuffles**: 2,666 (Score: 22.4/30, Quantitative Total: 32.4/40)
  - *Observation*: Achieves performance extremely close to `AnalyticalRolloutStrategy`. Weighting by burial depth is theoretically sound but shows minimal marginal improvement because most stack heights are kept low (2-3 containers) by the baseline height penalties, reducing the occurrence of deep burials.

### 8.6 FutureReservationStrategy
- **Description**: Accesses vessel schedule foreknowledge by pre-reserving empty/homogeneous stacks in the assigned block 4–12 hours before a vessel's discharge start window. Incoming containers for other vessels are penalized (+15.0 score penalty, -15.0 homogeneity penalty) from using these reserved stacks.
- **Results**:
  - **Train Reshuffles**: 2,873 (Score: 22.2/30, Quantitative Total: 32.2/40)
  - **Test Reshuffles**: 2,658 (Score: 22.5/30, Quantitative Total: 32.5/40)
  - *Observation*: Slightly worse on train, and very close to rollout on test. Reserving slots reduces the immediate choice set for other vessels sharing the same block, causing them to make suboptimal choices elsewhere, confirming that spatial restrictions often degrade online scheduling flexibility.

### 8.7 MLScorerStrategy
- **Description**: Trains an offline scikit-learn `GradientBoostingClassifier` (AUC **0.8125**) to predict reshuffle probability from 18 structural/temporal features.
- **Results**: Train 3,089 (31.3/40), Test 4,546 (24.1/40).
- *Observation*: High offline predictive power does not translate to online performance due to covariate shift — the model was trained on yard states created by the heuristic strategy.

### 8.8 PortRowPreferenceStrategy
- **Description**: Assigns soft row-level affinity based on destination ports (`preferred_row = (port_rank % rows) + 1`) to group same-destination containers along specific rows, creating dedicated "channels" to prevent cross-port contamination within a shared vessel block.
- **Results**:
  - **Train Reshuffles**: 2,913 (Score: 22.1/30, Quantitative Total: 32.1/40)
  - **Test Reshuffles**: 2,693 (Score: 22.3/30, Quantitative Total: 32.3/40)
  - *Observation*: Slightly worse than rollout. Row-level partitioning restricts flexibility and leads to overflow across rows during high-density port arrivals, confirming that soft zoning preferences still introduce minor coordination inefficiencies.

### 8.9 PlaceHijackingStrategy
- **Description**: Tracks newly exposed stack tops during container retrieval events. It queues a consolidation preference: the next time a container for the matching `(vessel, port)` arrives, it is directed to place on that exposed stack, virtually simulating re-marshalling.
- **Results**:
  - **Train Reshuffles**: 2,841 (Score: 22.4/30, Quantitative Total: 32.4/40)
  - **Test Reshuffles**: 2,686 (Score: 22.4/30, Quantitative Total: 32.4/40)
  - *Observation*: Improves upon baseline vessel pre-assignment but slightly underperforms analytical rollout. Consolidating stack tops after retrievals is highly effective for stack purity, but occasionally limits flexibility when the targeted stack is not structurally ideal.

### 8.10 LRK-Proximity Clustering (Final Production Strategy — `MyStrategy`)

- **Description**: Builds on `AnalyticalRolloutStrategy` with two key additions:
  1. **Exact retrieval times**: Pre-loads `LOAD` / `TRUCK_DLVR` timestamps from `events.jsonl` so LRK uses actual departure times (eliminating the import 10-day offset heuristic).
  2. **LRK-Proximity ERC-0 tiebreaker**: Among ERC=0 positions, prefer stacks whose top container has the closest retrieval time. Production weights: proximity **80.0** (`LRK_PROX_WEIGHT`), height **0.5** (`ERC0_HEIGHT_WEIGHT`), empty-stack bonus **−3.0**.
- **Results**: Train **2,240** (34.9/40), Test **2,367** (33.8/40).
- *Observation*: Since 99.98% of placements are ERC=0, the tiebreaker dominates decisions. Temporal clustering reduces cascade reshuffles from the simulator's LRK-unaware reshuffle placement. Reshuffles dropped **19%** on train (2,763→2,240) and **10.5%** on test (2,644→2,367) vs AnalyticalRolloutStrategy.

### 8.11 Future-Aware Placement (Tested & Rejected)

- **Description**: Pre-indexed all future placement events, computed exact future burial risk (containers with smaller LRK arriving before our retrieval), and targeted ideal stack tiers within each vessel's loading sequence.
- **Results**: Train 2,745–2,799 (32.8/40), vs **2,240** (34.9/40) without future-aware logic.
- *Observation*: The diagnosis was correct (future arrivals cause ERC=0 placements to become dirty), but the vessel-level penalty conflicted with LRK-proximity clustering already grouping containers by departure time. Ideal-tier targeting also fought the proximity signal. Not included in the final submission.

### 8.12 Global Multi-Block ERC-0 Search (Tested & Rejected)

- **Description**: Searched all 10 blocks simultaneously for ERC=0 positions with a $+5.0$ block-affinity penalty on non-primary blocks.
- **Results**: Did not improve the quantitative score over single-block search with the final weight configuration.
- *Observation*: Global search helped in early LRK-proximity experiments but the final single-block configuration scores highest. Not included in the final submission.

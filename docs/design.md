# Container Yard Placement Optimizer — Design Document

## 1. Problem Statement

In maritime and intermodal container terminals, efficient space utilization and crane productivity are major operational drivers. When containers arrive at the terminal (either from ships or trucks), the terminal operating system must decide where to stack them. Later, when these containers depart, they must be retrieved. If a target container is stacked under other containers, the ones on top must be temporarily relocated (reshuffled). Each reshuffle consumes crane time, delays ship loading, and increases operational costs.

This project implements a high-performance **hybrid container placement algorithm** that minimizes reshuffles by predicting retrieval order using multi-attribute signals (departure time, port of discharge, and weight class) and optimizing physical stack layouts in real time.

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
   - **Adaptation**: Used to apply a **truck uncertainty multiplier** (1.2x) to ERC for truck arrivals (`TRUCK_RECV`), reflecting the higher variability of truck arrivals compared to ships.

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

### 3.3 Stack Scoring Function

Each candidate stack in the target block is scored using a multi-attribute weighted function (lower score is better):
$$\text{Score} = \alpha \cdot \text{ERC}_{\text{eff}} + \beta \cdot \text{Height Penalty} + \gamma \cdot \text{Neighborhood Penalty} + \zeta \cdot \text{Block Penalty} - \delta \cdot \text{Cohesion Bonus}$$
Where:
- $\text{ERC}_{\text{eff}} = \text{ERC} \times 1.2$ for `TRUCK_RECV` and $\text{ERC} \times 1.0$ for `DISCHARGE`.
- **Height Penalty**: $\text{height}^2 \times 0.15 \times \text{occupancy}$. Penalizes tall stacks to keep stack heights balanced, preserving relocation capacity.
- **Neighborhood Penalty**: Penalizes stacks that are adjacent to much taller stacks ($&gt;1$ tier height difference) to prevent building "walls" that block crane access.
- **Cohesion Bonus**: $+2.0$ for same vessel, $+1.5$ for same port, and $+0.5$ per adjacent same-vessel container. Groups same-destination cargo.
- **Adaptive Block Penalty**: Penalizes blocks experiencing high historical reshuffle rates.
- **Dynamic Weights**: $\alpha$ (ERC weight) increases from 10 to 15, and $\delta$ (cohesion bonus) decreases from 3.0 to 1.5 when yard occupancy exceeds 80%.

### 3.4 Decision Flow

```
   Incoming Event (DISCHARGE or TRUCK_RECV)
                      │
            ┌─────────▼─────────┐
            │   Block Selection │  Pin vessel to preferred block (based on ETD bucket)
            └─────────┬─────────┘
                      │
            ┌─────────▼─────────┐
            │   Stack Filtering │  Filter available stacks using self.non_full_stacks cache
            └─────────┬─────────┘  (safety height buffer = 4 tiers)
                      │
            ┌─────────▼─────────┐
            │    Stack Scoring  │  Score each stack in target block. Pick minimum score.
            └─────────┬─────────┘
                      │
            ┌─────────▼─────────┐
            │     Fallback      │  If block full, try other blocks in priority.
            └───────────────────┘  If all full, run global greedy lowest-stack.
```

---

## 4. Train Data Analysis Summary

An analysis of the training dataset (`data/train`) revealed key structural properties:
- **Event Distribution**: `DISCHARGE` (7,985) and `LOAD` (7,719) make up 76.3% of events. `TRUCK_RECV` (2,400) and `TRUCK_DLVR` (2,488) make up the remaining 23.7%.
- **Vessel Volatility**: 20 distinct vessels. `VSL003` dominates with 1,657 discharge events (16% of total placements).
- **Time Span**: The simulation runs over 33.1 days.
- **Alphabetical Port Ordering**: Analysis confirmed that for *every* vessel rotation, the loading sequence of port groups is strictly alphabetical. This validates our sorting rank logic.
- **Initial State Density**: The initial state contains 4,800 containers, placing the yard at 50% occupancy at start. These initial stacks were generated by a greedy baseline, meaning the yard starts in a highly disorganized state.

---

## 5. Performance Comparison & Quantitative Score

The table below compares the performance of our hybrid strategy against the two baselines on the training and test datasets:

| Dataset | Metric | Random Baseline | Greedy Baseline | **Our Hybrid Strategy** |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | Total Reshuffles | 8,933 | 8,036 | **3,757** |
| | Reshuffles/Retrieval | 0.8752 | 0.7873 | **0.3681** |
| | Score — Reshuffles | 0.0 / 30.0 | 0.5 / 30.0 | **18.5 / 30.0** |
| | **Quantitative Total** | **10.0 / 40.0** | **10.5 / 40.0** | **28.5 / 40.0** |
| **Test** | Total Reshuffles | 9,122* | 7,288* | **3,878** |
| | Reshuffles/Retrieval | 0.8883* | 0.7100* | **0.4020** |
| | Score — Reshuffles | 0.0 / 30.0 | 3.9 / 30.0 | **17.1 / 30.0** |
| | **Quantitative Total** | **10.0 / 40.0** | **13.9 / 40.0** | **27.1 / 40.0** |

*\*Note: Baseline scores on the test set are referenced from `src/scoring.py`.*

Our hybrid strategy achieved a **~50% reduction in reshuffles** compared to the greedy baseline on both datasets, yielding a quantitative score of **27.1 / 40.0** on the test dataset.

---

## 6. Time and Space Complexity

- **Time Complexity per Placement**:
  - Block Selection: $O(1)$ block lookups.
  - Stack Selection: We iterate over the cached non-full stacks within the target block. A block has at most $B \times R = 240$ stacks. For each stack, evaluating the score takes $O(\text{height}) \le O(5) = O(1)$.
  - Total Time: $O(B \times R) \approx 240$ operations. In Python, this evaluates in under **1 millisecond** per event, processing 20,000 events in 3.8 seconds.
- **Space Complexity**:
  - We store block layouts, vessel schedules, block assignment caches, and non-full stack sets.
  - Total Space: $O(V + B \times R) \approx O(20 \text{ vessels} + 10 \text{ blocks} \times 240 \text{ stacks}) = O(2,400)$, requiring less than **2 MB** of memory.

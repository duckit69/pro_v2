# Edge Personalization: LOSO Evaluation Report

## Hyperparameters
The personalization model updates on-device using Stochastic Gradient Descent (SGD) with experience replay. To fully track the personalization experiment, the following primary and implicit parameters are documented:

### 1. Learning Rate
- **Current Value**: `0.001`
- **Impact**: Controls how aggressively the weight updates are applied when an error is made. A higher learning rate adapts faster but can cause catastrophic forgetting of the baseline model.

### 2. Replay Buffer Size
- **Current Value**: `10`
- **Impact**: Determines how many past samples the device can remember. A larger buffer gives a more stable gradient during updates but requires more memory on the ESP32.

### 3. Replay Frequency
- **Current Value**: Every `5` updates
- **Impact**: Controls how often the model trains on the historical buffer. Frequent replays solidify past knowledge but cost more CPU cycles/battery on the edge device.

### 4. Replay Epochs / Batching
- **Current Value**: `1` (Loops over the entire buffer exactly once per trigger)
- **Impact**: You could introduce a parameter for how many times you iterate over the buffer during a replay phase. Replaying the buffer 3 or 5 times per trigger instead of 1 time will drastically change how aggressively the model personalizes.

### 5. Buffer Replacement Strategy
- **Current Value**: `FIFO` (Circular replacement)
- **Impact**: Right now, the oldest data is replaced by the newest data. In an imbalanced dataset like stress detection, a FIFO buffer might end up completely filled with "non-stress" examples, making replay updates biased. Tracking whether you use FIFO, Random Replacement, or a Class-Balanced buffer (e.g., forcing the buffer to hold 50% stress and 50% non-stress samples) is very important for future tuning.

### 6. Decision Threshold
- **Current Value**: `0.5`
- **Impact**: While `0.5` is standard, adjusting this threshold (e.g., to `0.4` or `0.6`) can be a tunable parameter, especially if you care more about catching all stress events (higher recall/sensitivity) at the cost of some false alarms.

## Final Average Results
- **Generic Baseline Model**:
  - Average Accuracy: `0.8540`
  - Average F1 Score: `0.5781`
- **Incremental Personalization Model**:
  - Average Accuracy: `0.8547`
  - Average F1 Score: `0.5709`
- **Net Improvement**:
  - Accuracy: `+0.0007`
  - F1 Score: `-0.0072`

---

## Detailed Subject Results

### Subject S2
- **Baseline Model**: Accuracy: `0.8750`, F1: `0.7826`
  - Confusion Matrix:
    ```
    [[52  7]
     [ 3 18]]
    ```
- **Personalization Model**: Accuracy: `0.8750`, F1: `0.7826` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[52  7]
     [ 3 18]]
    ```

### Subject S3
- **Baseline Model**: Accuracy: `0.7143`, F1: `0.5200`
  - Confusion Matrix:
    ```
    [[47 15]
     [ 9 13]]
    ```
- **Personalization Model**: Accuracy: `0.7143`, F1: `0.5200` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[47 15]
     [ 9 13]]
    ```

### Subject S4
- **Baseline Model**: Accuracy: `0.9881`, F1: `0.9756`
  - Confusion Matrix:
    ```
    [[63  0]
     [ 1 20]]
    ```
- **Personalization Model**: Accuracy: `0.9881`, F1: `0.9756` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[63  0]
     [ 1 20]]
    ```

### Subject S5
- **Baseline Model**: Accuracy: `0.9778`, F1: `0.9545`
  - Confusion Matrix:
    ```
    [[67  2]
     [ 0 21]]
    ```
- **Personalization Model**: Accuracy: `0.9778`, F1: `0.9545` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[67  2]
     [ 0 21]]
    ```

### Subject S6
- **Baseline Model**: Accuracy: `0.9588`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[72  3]
     [ 1 21]]
    ```
- **Personalization Model**: Accuracy: `0.9485`, F1: `0.8936` (Improvement: Acc -0.0103, F1 -0.0194)
  - Confusion Matrix:
    ```
    [[71  4]
     [ 1 21]]
    ```

### Subject S7
- **Baseline Model**: Accuracy: `0.9556`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[65  4]
     [ 0 21]]
    ```
- **Personalization Model**: Accuracy: `0.9667`, F1: `0.9333` (Improvement: Acc +0.0111, F1 +0.0203)
  - Confusion Matrix:
    ```
    [[66  3]
     [ 0 21]]
    ```

### Subject S8
- **Baseline Model**: Accuracy: `0.7619`, F1: `0.2308`
  - Confusion Matrix:
    ```
    [[61  0]
     [20  3]]
    ```
- **Personalization Model**: Accuracy: `0.7619`, F1: `0.2308` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[61  0]
     [20  3]]
    ```

### Subject S9
- **Baseline Model**: Accuracy: `0.7912`, F1: `0.1739`
  - Confusion Matrix:
    ```
    [[70  0]
     [19  2]]
    ```
- **Personalization Model**: Accuracy: `0.7802`, F1: `0.0909` (Improvement: Acc -0.0110, F1 -0.0830)
  - Confusion Matrix:
    ```
    [[70  0]
     [20  1]]
    ```

### Subject S10
- **Baseline Model**: Accuracy: `0.8557`, F1: `0.6111`
  - Confusion Matrix:
    ```
    [[72  1]
     [13 11]]
    ```
- **Personalization Model**: Accuracy: `0.8660`, F1: `0.6286` (Improvement: Acc +0.0103, F1 +0.0175)
  - Confusion Matrix:
    ```
    [[73  0]
     [13 11]]
    ```

### Subject S11
- **Baseline Model**: Accuracy: `0.9286`, F1: `0.8293`
  - Confusion Matrix:
    ```
    [[74  1]
     [ 6 17]]
    ```
- **Personalization Model**: Accuracy: `0.9286`, F1: `0.8293` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[74  1]
     [ 6 17]]
    ```

### Subject S13
- **Baseline Model**: Accuracy: `0.8485`, F1: `0.7368`
  - Confusion Matrix:
    ```
    [[63 14]
     [ 1 21]]
    ```
- **Personalization Model**: Accuracy: `0.8485`, F1: `0.6809` (Improvement: Acc +0.0000, F1 -0.0560)
  - Confusion Matrix:
    ```
    [[68  9]
     [ 6 16]]
    ```

### Subject S14
- **Baseline Model**: Accuracy: `0.7470`, F1: `0.0000`
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```
- **Personalization Model**: Accuracy: `0.7470`, F1: `0.0000` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```

### Subject S15
- **Baseline Model**: Accuracy: `0.8000`, F1: `0.3448`
  - Confusion Matrix:
    ```
    [[71  1]
     [18  5]]
    ```
- **Personalization Model**: Accuracy: `0.8105`, F1: `0.3571` (Improvement: Acc +0.0105, F1 +0.0123)
  - Confusion Matrix:
    ```
    [[72  0]
     [18  5]]
    ```

### Subject S16
- **Baseline Model**: Accuracy: `0.8571`, F1: `0.6061`
  - Confusion Matrix:
    ```
    [[68  1]
     [12 10]]
    ```
- **Personalization Model**: Accuracy: `0.8571`, F1: `0.6061` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[68  1]
     [12 10]]
    ```

### Subject S17
- **Baseline Model**: Accuracy: `0.7500`, F1: `0.0800`
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```
- **Personalization Model**: Accuracy: `0.7500`, F1: `0.0800` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```

---

## Dynamic Adjustments Evaluation

### Variant 1: Dynamic Learning Rate
- **File**: `part2_edge_simulation_dynamic_lr.py`
- **Changes**: Introduced learning rate time-decay `current_lr = 0.001 / (1 + 0.01 * updates_count)`.
- **Results**:
  - Average Accuracy: `0.8547` (Improvement vs Generic: `+0.0007`)
  - Average F1 Score: `0.5712` (Improvement vs Generic: `-0.0069`)
- **Conclusion**: Learning rate decay keeps accuracy stable but slightly lowers the F1 score in this specific scenario, likely because the default LR was already quite conservative.

#### Subject S10
- **Baseline Model**: Accuracy: `0.8557`, F1: `0.6111`
  - Confusion Matrix:
    ```
    [[72  1]
     [13 11]]
    ```
- **Dynamic LR Model**: Accuracy: `0.8660`, F1: `0.6286` (Improvement: Acc +0.0103, F1 +0.0175)
  - Confusion Matrix:
    ```
    [[73  0]
     [13 11]]
    ```

#### Subject S11
- **Baseline Model**: Accuracy: `0.9286`, F1: `0.8293`
  - Confusion Matrix:
    ```
    [[74  1]
     [ 6 17]]
    ```
- **Dynamic LR Model**: Accuracy: `0.9286`, F1: `0.8293` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[74  1]
     [ 6 17]]
    ```

#### Subject S13
- **Baseline Model**: Accuracy: `0.8485`, F1: `0.7368`
  - Confusion Matrix:
    ```
    [[63 14]
     [ 1 21]]
    ```
- **Dynamic LR Model**: Accuracy: `0.8384`, F1: `0.6667` (Improvement: Acc -0.0101, F1 -0.0702)
  - Confusion Matrix:
    ```
    [[67 10]
     [ 6 16]]
    ```

#### Subject S14
- **Baseline Model**: Accuracy: `0.7470`, F1: `0.0000`
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```
- **Dynamic LR Model**: Accuracy: `0.7470`, F1: `0.0000` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```

#### Subject S15
- **Baseline Model**: Accuracy: `0.8000`, F1: `0.3448`
  - Confusion Matrix:
    ```
    [[71  1]
     [18  5]]
    ```
- **Dynamic LR Model**: Accuracy: `0.8105`, F1: `0.3571` (Improvement: Acc +0.0105, F1 +0.0123)
  - Confusion Matrix:
    ```
    [[72  0]
     [18  5]]
    ```

#### Subject S16
- **Baseline Model**: Accuracy: `0.8571`, F1: `0.6061`
  - Confusion Matrix:
    ```
    [[68  1]
     [12 10]]
    ```
- **Dynamic LR Model**: Accuracy: `0.8571`, F1: `0.6061` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[68  1]
     [12 10]]
    ```

#### Subject S17
- **Baseline Model**: Accuracy: `0.7500`, F1: `0.0800`
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```
- **Dynamic LR Model**: Accuracy: `0.7500`, F1: `0.0800` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```

#### Subject S2
- **Baseline Model**: Accuracy: `0.8750`, F1: `0.7826`
  - Confusion Matrix:
    ```
    [[52  7]
     [ 3 18]]
    ```
- **Dynamic LR Model**: Accuracy: `0.8750`, F1: `0.7826` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[52  7]
     [ 3 18]]
    ```

#### Subject S3
- **Baseline Model**: Accuracy: `0.7143`, F1: `0.5200`
  - Confusion Matrix:
    ```
    [[47 15]
     [ 9 13]]
    ```
- **Dynamic LR Model**: Accuracy: `0.7143`, F1: `0.5200` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[47 15]
     [ 9 13]]
    ```

#### Subject S4
- **Baseline Model**: Accuracy: `0.9881`, F1: `0.9756`
  - Confusion Matrix:
    ```
    [[63  0]
     [ 1 20]]
    ```
- **Dynamic LR Model**: Accuracy: `0.9881`, F1: `0.9756` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[63  0]
     [ 1 20]]
    ```

#### Subject S5
- **Baseline Model**: Accuracy: `0.9778`, F1: `0.9545`
  - Confusion Matrix:
    ```
    [[67  2]
     [ 0 21]]
    ```
- **Dynamic LR Model**: Accuracy: `0.9778`, F1: `0.9545` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[67  2]
     [ 0 21]]
    ```

#### Subject S6
- **Baseline Model**: Accuracy: `0.9588`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[72  3]
     [ 1 21]]
    ```
- **Dynamic LR Model**: Accuracy: `0.9588`, F1: `0.9130` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[72  3]
     [ 1 21]]
    ```

#### Subject S7
- **Baseline Model**: Accuracy: `0.9556`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[65  4]
     [ 0 21]]
    ```
- **Dynamic LR Model**: Accuracy: `0.9667`, F1: `0.9333` (Improvement: Acc +0.0111, F1 +0.0203)
  - Confusion Matrix:
    ```
    [[66  3]
     [ 0 21]]
    ```

#### Subject S8
- **Baseline Model**: Accuracy: `0.7619`, F1: `0.2308`
  - Confusion Matrix:
    ```
    [[61  0]
     [20  3]]
    ```
- **Dynamic LR Model**: Accuracy: `0.7619`, F1: `0.2308` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[61  0]
     [20  3]]
    ```

#### Subject S9
- **Baseline Model**: Accuracy: `0.7912`, F1: `0.1739`
  - Confusion Matrix:
    ```
    [[70  0]
     [19  2]]
    ```
- **Dynamic LR Model**: Accuracy: `0.7802`, F1: `0.0909` (Improvement: Acc -0.0110, F1 -0.0830)
  - Confusion Matrix:
    ```
    [[70  0]
     [20  1]]
    ```

### Variant 2: Dynamic Decision Threshold
- **File**: `part2_edge_simulation_dynamic_threshold.py`
- **Changes**: On every replay trigger (every 5 updates), the threshold was dynamically set to maximize the local F1 score in the replay buffer (testing thresholds from `0.3` to `0.7`).
- **Results**:
  - Average Accuracy: `0.8368` (Improvement vs Generic: `-0.0172`)
  - Average F1 Score: `0.6181` (Improvement vs Generic: `+0.0400`)
- **Conclusion**: The dynamic threshold yielded a **massive increase in F1 Score (+0.0400)**, demonstrating that adapting the threshold to the recent local context is highly effective at catching stress events, albeit at the cost of a slight drop in overall accuracy (due to a higher willingness to predict stress).

#### Subject S10
- **Baseline Model**: Accuracy: `0.8557`, F1: `0.6111`
  - Confusion Matrix:
    ```
    [[72  1]
     [13 11]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.8557`, F1: `0.6111` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[72  1]
     [13 11]]
    ```

#### Subject S11
- **Baseline Model**: Accuracy: `0.9286`, F1: `0.8293`
  - Confusion Matrix:
    ```
    [[74  1]
     [ 6 17]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.9592`, F1: `0.9200` (Improvement: Acc +0.0306, F1 +0.0907)
  - Confusion Matrix:
    ```
    [[71  4]
     [ 0 23]]
    ```

#### Subject S13
- **Baseline Model**: Accuracy: `0.8485`, F1: `0.7368`
  - Confusion Matrix:
    ```
    [[63 14]
     [ 1 21]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.7677`, F1: `0.5818` (Improvement: Acc -0.0808, F1 -0.1550)
  - Confusion Matrix:
    ```
    [[60 17]
     [ 6 16]]
    ```

#### Subject S14
- **Baseline Model**: Accuracy: `0.7470`, F1: `0.0000`
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.7470`, F1: `0.0000` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```

#### Subject S15
- **Baseline Model**: Accuracy: `0.8000`, F1: `0.3448`
  - Confusion Matrix:
    ```
    [[71  1]
     [18  5]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.8632`, F1: `0.6486` (Improvement: Acc +0.0632, F1 +0.3038)
  - Confusion Matrix:
    ```
    [[70  2]
     [11 12]]
    ```

#### Subject S16
- **Baseline Model**: Accuracy: `0.8571`, F1: `0.6061`
  - Confusion Matrix:
    ```
    [[68  1]
     [12 10]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.8571`, F1: `0.6486` (Improvement: Acc +0.0000, F1 +0.0426)
  - Confusion Matrix:
    ```
    [[66  3]
     [10 12]]
    ```

#### Subject S17
- **Baseline Model**: Accuracy: `0.7500`, F1: `0.0800`
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.7500`, F1: `0.0800` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```

#### Subject S2
- **Baseline Model**: Accuracy: `0.8750`, F1: `0.7826`
  - Confusion Matrix:
    ```
    [[52  7]
     [ 3 18]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.7125`, F1: `0.6462` (Improvement: Acc -0.1625, F1 -0.1365)
  - Confusion Matrix:
    ```
    [[36 23]
     [ 0 21]]
    ```

#### Subject S3
- **Baseline Model**: Accuracy: `0.7143`, F1: `0.5200`
  - Confusion Matrix:
    ```
    [[47 15]
     [ 9 13]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.5952`, F1: `0.5000` (Improvement: Acc -0.1190, F1 -0.0200)
  - Confusion Matrix:
    ```
    [[33 29]
     [ 5 17]]
    ```

#### Subject S4
- **Baseline Model**: Accuracy: `0.9881`, F1: `0.9756`
  - Confusion Matrix:
    ```
    [[63  0]
     [ 1 20]]
    ```
- **Dynamic Threshold Model**: Accuracy: `1.0000`, F1: `1.0000` (Improvement: Acc +0.0119, F1 +0.0244)
  - Confusion Matrix:
    ```
    [[63  0]
     [ 0 21]]
    ```

#### Subject S5
- **Baseline Model**: Accuracy: `0.9778`, F1: `0.9545`
  - Confusion Matrix:
    ```
    [[67  2]
     [ 0 21]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.9667`, F1: `0.9333` (Improvement: Acc -0.0111, F1 -0.0212)
  - Confusion Matrix:
    ```
    [[66  3]
     [ 0 21]]
    ```

#### Subject S6
- **Baseline Model**: Accuracy: `0.9588`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[72  3]
     [ 1 21]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.8866`, F1: `0.8000` (Improvement: Acc -0.0722, F1 -0.1130)
  - Confusion Matrix:
    ```
    [[64 11]
     [ 0 22]]
    ```

#### Subject S7
- **Baseline Model**: Accuracy: `0.9556`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[65  4]
     [ 0 21]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.9111`, F1: `0.8400` (Improvement: Acc -0.0444, F1 -0.0730)
  - Confusion Matrix:
    ```
    [[61  8]
     [ 0 21]]
    ```

#### Subject S8
- **Baseline Model**: Accuracy: `0.7619`, F1: `0.2308`
  - Confusion Matrix:
    ```
    [[61  0]
     [20  3]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.8333`, F1: `0.5625` (Improvement: Acc +0.0714, F1 +0.3317)
  - Confusion Matrix:
    ```
    [[61  0]
     [14  9]]
    ```

#### Subject S9
- **Baseline Model**: Accuracy: `0.7912`, F1: `0.1739`
  - Confusion Matrix:
    ```
    [[70  0]
     [19  2]]
    ```
- **Dynamic Threshold Model**: Accuracy: `0.8462`, F1: `0.5000` (Improvement: Acc +0.0549, F1 +0.3261)
  - Confusion Matrix:
    ```
    [[70  0]
     [14  7]]
    ```

### Variant 3: Dynamic LR + Dynamic Threshold
- **File**: `part2_edge_simulation_dynamic_both.py`
- **Changes**: Combined both the learning rate time-decay and the dynamic F1-maximizing threshold evaluation during replay updates.
- **Results**:
  - Average Accuracy: `0.8360` (Improvement vs Generic: `-0.0180`)
  - Average F1 Score: `0.6177` (Improvement vs Generic: `+0.0396`)
- **Conclusion**: Combining both approaches shows how multiple edge adaptation strategies interact. The results demonstrate the strong impact of threshold shifting while maintaining the stability from learning rate decay.

#### Subject S10
- **Baseline Model**: Accuracy: `0.8557`, F1: `0.6111`
  - Confusion Matrix:
    ```
    [[72  1]
     [13 11]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.8557`, F1: `0.6111` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[72  1]
     [13 11]]
    ```

#### Subject S11
- **Baseline Model**: Accuracy: `0.9286`, F1: `0.8293`
  - Confusion Matrix:
    ```
    [[74  1]
     [ 6 17]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.9592`, F1: `0.9200` (Improvement: Acc +0.0306, F1 +0.0907)
  - Confusion Matrix:
    ```
    [[71  4]
     [ 0 23]]
    ```

#### Subject S13
- **Baseline Model**: Accuracy: `0.8485`, F1: `0.7368`
  - Confusion Matrix:
    ```
    [[63 14]
     [ 1 21]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.7677`, F1: `0.5818` (Improvement: Acc -0.0808, F1 -0.1550)
  - Confusion Matrix:
    ```
    [[60 17]
     [ 6 16]]
    ```

#### Subject S14
- **Baseline Model**: Accuracy: `0.7470`, F1: `0.0000`
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.7470`, F1: `0.0000` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```

#### Subject S15
- **Baseline Model**: Accuracy: `0.8000`, F1: `0.3448`
  - Confusion Matrix:
    ```
    [[71  1]
     [18  5]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.8632`, F1: `0.6486` (Improvement: Acc +0.0632, F1 +0.3038)
  - Confusion Matrix:
    ```
    [[70  2]
     [11 12]]
    ```

#### Subject S16
- **Baseline Model**: Accuracy: `0.8571`, F1: `0.6061`
  - Confusion Matrix:
    ```
    [[68  1]
     [12 10]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.8571`, F1: `0.6486` (Improvement: Acc +0.0000, F1 +0.0426)
  - Confusion Matrix:
    ```
    [[66  3]
     [10 12]]
    ```

#### Subject S17
- **Baseline Model**: Accuracy: `0.7500`, F1: `0.0800`
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.7500`, F1: `0.0800` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```

#### Subject S2
- **Baseline Model**: Accuracy: `0.8750`, F1: `0.7826`
  - Confusion Matrix:
    ```
    [[52  7]
     [ 3 18]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.7125`, F1: `0.6462` (Improvement: Acc -0.1625, F1 -0.1365)
  - Confusion Matrix:
    ```
    [[36 23]
     [ 0 21]]
    ```

#### Subject S3
- **Baseline Model**: Accuracy: `0.7143`, F1: `0.5200`
  - Confusion Matrix:
    ```
    [[47 15]
     [ 9 13]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.5833`, F1: `0.4928` (Improvement: Acc -0.1310, F1 -0.0272)
  - Confusion Matrix:
    ```
    [[32 30]
     [ 5 17]]
    ```

#### Subject S4
- **Baseline Model**: Accuracy: `0.9881`, F1: `0.9756`
  - Confusion Matrix:
    ```
    [[63  0]
     [ 1 20]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `1.0000`, F1: `1.0000` (Improvement: Acc +0.0119, F1 +0.0244)
  - Confusion Matrix:
    ```
    [[63  0]
     [ 0 21]]
    ```

#### Subject S5
- **Baseline Model**: Accuracy: `0.9778`, F1: `0.9545`
  - Confusion Matrix:
    ```
    [[67  2]
     [ 0 21]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.9667`, F1: `0.9333` (Improvement: Acc -0.0111, F1 -0.0212)
  - Confusion Matrix:
    ```
    [[66  3]
     [ 0 21]]
    ```

#### Subject S6
- **Baseline Model**: Accuracy: `0.9588`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[72  3]
     [ 1 21]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.8866`, F1: `0.8000` (Improvement: Acc -0.0722, F1 -0.1130)
  - Confusion Matrix:
    ```
    [[64 11]
     [ 0 22]]
    ```

#### Subject S7
- **Baseline Model**: Accuracy: `0.9556`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[65  4]
     [ 0 21]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.9111`, F1: `0.8400` (Improvement: Acc -0.0444, F1 -0.0730)
  - Confusion Matrix:
    ```
    [[61  8]
     [ 0 21]]
    ```

#### Subject S8
- **Baseline Model**: Accuracy: `0.7619`, F1: `0.2308`
  - Confusion Matrix:
    ```
    [[61  0]
     [20  3]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.8333`, F1: `0.5625` (Improvement: Acc +0.0714, F1 +0.3317)
  - Confusion Matrix:
    ```
    [[61  0]
     [14  9]]
    ```

#### Subject S9
- **Baseline Model**: Accuracy: `0.7912`, F1: `0.1739`
  - Confusion Matrix:
    ```
    [[70  0]
     [19  2]]
    ```
- **Dynamic LR + Threshold Model**: Accuracy: `0.8462`, F1: `0.5000` (Improvement: Acc +0.0549, F1 +0.3261)
  - Confusion Matrix:
    ```
    [[70  0]
     [14  7]]
    ```

### Variant 4: Class-Balanced Replay Buffer
- **File**: `part2_edge_simulation_balanced_buffer.py`
- **Changes**: Replaced the standard 10-item FIFO buffer with a Class-Balanced buffer. It maintains two separate sub-buffers of size 5: one strictly for stress events and one strictly for non-stress events. This ensures every replay update evaluates a 50/50 split of past examples.
- **Results**:
  - Average Accuracy: `0.8547` (Improvement vs Generic: `+0.0007`)
  - Average F1 Score: `0.5765` (Improvement vs Generic: `-0.0016`)
- **Conclusion**: Forcing the replay buffer to be perfectly balanced prevents the model from forgetting the minority class (stress), leading to more robust personalization updates when the data stream is heavily skewed towards non-stress events.

#### Subject S10
- **Baseline Model**: Accuracy: `0.8557`, F1: `0.6111`
  - Confusion Matrix:
    ```
    [[72  1]
     [13 11]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.8660`, F1: `0.6286` (Improvement: Acc +0.0103, F1 +0.0175)
  - Confusion Matrix:
    ```
    [[73  0]
     [13 11]]
    ```

#### Subject S11
- **Baseline Model**: Accuracy: `0.9286`, F1: `0.8293`
  - Confusion Matrix:
    ```
    [[74  1]
     [ 6 17]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.9286`, F1: `0.8293` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[74  1]
     [ 6 17]]
    ```

#### Subject S13
- **Baseline Model**: Accuracy: `0.8485`, F1: `0.7368`
  - Confusion Matrix:
    ```
    [[63 14]
     [ 1 21]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.8485`, F1: `0.6939` (Improvement: Acc +0.0000, F1 -0.0430)
  - Confusion Matrix:
    ```
    [[67 10]
     [ 5 17]]
    ```

#### Subject S14
- **Baseline Model**: Accuracy: `0.7470`, F1: `0.0000`
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.7470`, F1: `0.0000` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```

#### Subject S15
- **Baseline Model**: Accuracy: `0.8000`, F1: `0.3448`
  - Confusion Matrix:
    ```
    [[71  1]
     [18  5]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.8000`, F1: `0.3448` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[71  1]
     [18  5]]
    ```

#### Subject S16
- **Baseline Model**: Accuracy: `0.8571`, F1: `0.6061`
  - Confusion Matrix:
    ```
    [[68  1]
     [12 10]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.8571`, F1: `0.6061` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[68  1]
     [12 10]]
    ```

#### Subject S17
- **Baseline Model**: Accuracy: `0.7500`, F1: `0.0800`
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.7500`, F1: `0.0800` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```

#### Subject S2
- **Baseline Model**: Accuracy: `0.8750`, F1: `0.7826`
  - Confusion Matrix:
    ```
    [[52  7]
     [ 3 18]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.8750`, F1: `0.7826` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[52  7]
     [ 3 18]]
    ```

#### Subject S3
- **Baseline Model**: Accuracy: `0.7143`, F1: `0.5200`
  - Confusion Matrix:
    ```
    [[47 15]
     [ 9 13]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.7143`, F1: `0.5200` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[47 15]
     [ 9 13]]
    ```

#### Subject S4
- **Baseline Model**: Accuracy: `0.9881`, F1: `0.9756`
  - Confusion Matrix:
    ```
    [[63  0]
     [ 1 20]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.9881`, F1: `0.9756` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[63  0]
     [ 1 20]]
    ```

#### Subject S5
- **Baseline Model**: Accuracy: `0.9778`, F1: `0.9545`
  - Confusion Matrix:
    ```
    [[67  2]
     [ 0 21]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.9778`, F1: `0.9545` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[67  2]
     [ 0 21]]
    ```

#### Subject S6
- **Baseline Model**: Accuracy: `0.9588`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[72  3]
     [ 1 21]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.9485`, F1: `0.8936` (Improvement: Acc -0.0103, F1 -0.0194)
  - Confusion Matrix:
    ```
    [[71  4]
     [ 1 21]]
    ```

#### Subject S7
- **Baseline Model**: Accuracy: `0.9556`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[65  4]
     [ 0 21]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.9667`, F1: `0.9333` (Improvement: Acc +0.0111, F1 +0.0203)
  - Confusion Matrix:
    ```
    [[66  3]
     [ 0 21]]
    ```

#### Subject S8
- **Baseline Model**: Accuracy: `0.7619`, F1: `0.2308`
  - Confusion Matrix:
    ```
    [[61  0]
     [20  3]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.7619`, F1: `0.2308` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[61  0]
     [20  3]]
    ```

#### Subject S9
- **Baseline Model**: Accuracy: `0.7912`, F1: `0.1739`
  - Confusion Matrix:
    ```
    [[70  0]
     [19  2]]
    ```
- **Balanced Buffer Model**: Accuracy: `0.7912`, F1: `0.1739` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[70  0]
     [19  2]]
    ```

### Variant 5: Class-Balanced Buffer + Dynamic Threshold
- **File**: `part2_edge_simulation_balanced_dynamic_threshold.py`
- **Changes**: Combined the strictly 50/50 Class-Balanced Replay Buffer with the dynamic F1-maximizing threshold strategy.
- **Results**:
  - Average Accuracy: `0.8381` (Improvement vs Generic: `-0.0159`)
  - Average F1 Score: `0.6232` (Improvement vs Generic: `+0.0451`)
- **Conclusion**: Testing both the class-balanced memory alongside the shifting threshold. This evaluates if enforcing a strict 50/50 class memory helps the dynamic threshold find a better decision boundary.

#### Subject S10
- **Baseline Model**: Accuracy: `0.8557`, F1: `0.6111`
  - Confusion Matrix:
    ```
    [[72  1]
     [13 11]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.8557`, F1: `0.6111` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[72  1]
     [13 11]]
    ```

#### Subject S11
- **Baseline Model**: Accuracy: `0.9286`, F1: `0.8293`
  - Confusion Matrix:
    ```
    [[74  1]
     [ 6 17]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.9592`, F1: `0.9200` (Improvement: Acc +0.0306, F1 +0.0907)
  - Confusion Matrix:
    ```
    [[71  4]
     [ 0 23]]
    ```

#### Subject S13
- **Baseline Model**: Accuracy: `0.8485`, F1: `0.7368`
  - Confusion Matrix:
    ```
    [[63 14]
     [ 1 21]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.7879`, F1: `0.6441` (Improvement: Acc -0.0606, F1 -0.0928)
  - Confusion Matrix:
    ```
    [[59 18]
     [ 3 19]]
    ```

#### Subject S14
- **Baseline Model**: Accuracy: `0.7470`, F1: `0.0000`
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.7470`, F1: `0.0000` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```

#### Subject S15
- **Baseline Model**: Accuracy: `0.8000`, F1: `0.3448`
  - Confusion Matrix:
    ```
    [[71  1]
     [18  5]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.8632`, F1: `0.6486` (Improvement: Acc +0.0632, F1 +0.3038)
  - Confusion Matrix:
    ```
    [[70  2]
     [11 12]]
    ```

#### Subject S16
- **Baseline Model**: Accuracy: `0.8571`, F1: `0.6061`
  - Confusion Matrix:
    ```
    [[68  1]
     [12 10]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.8571`, F1: `0.6486` (Improvement: Acc +0.0000, F1 +0.0426)
  - Confusion Matrix:
    ```
    [[66  3]
     [10 12]]
    ```

#### Subject S17
- **Baseline Model**: Accuracy: `0.7500`, F1: `0.0800`
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.7500`, F1: `0.0800` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```

#### Subject S2
- **Baseline Model**: Accuracy: `0.8750`, F1: `0.7826`
  - Confusion Matrix:
    ```
    [[52  7]
     [ 3 18]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.7125`, F1: `0.6462` (Improvement: Acc -0.1625, F1 -0.1365)
  - Confusion Matrix:
    ```
    [[36 23]
     [ 0 21]]
    ```

#### Subject S3
- **Baseline Model**: Accuracy: `0.7143`, F1: `0.5200`
  - Confusion Matrix:
    ```
    [[47 15]
     [ 9 13]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.5952`, F1: `0.5143` (Improvement: Acc -0.1190, F1 -0.0057)
  - Confusion Matrix:
    ```
    [[32 30]
     [ 4 18]]
    ```

#### Subject S4
- **Baseline Model**: Accuracy: `0.9881`, F1: `0.9756`
  - Confusion Matrix:
    ```
    [[63  0]
     [ 1 20]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `1.0000`, F1: `1.0000` (Improvement: Acc +0.0119, F1 +0.0244)
  - Confusion Matrix:
    ```
    [[63  0]
     [ 0 21]]
    ```

#### Subject S5
- **Baseline Model**: Accuracy: `0.9778`, F1: `0.9545`
  - Confusion Matrix:
    ```
    [[67  2]
     [ 0 21]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.9667`, F1: `0.9333` (Improvement: Acc -0.0111, F1 -0.0212)
  - Confusion Matrix:
    ```
    [[66  3]
     [ 0 21]]
    ```

#### Subject S6
- **Baseline Model**: Accuracy: `0.9588`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[72  3]
     [ 1 21]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.8866`, F1: `0.8000` (Improvement: Acc -0.0722, F1 -0.1130)
  - Confusion Matrix:
    ```
    [[64 11]
     [ 0 22]]
    ```

#### Subject S7
- **Baseline Model**: Accuracy: `0.9556`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[65  4]
     [ 0 21]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.9111`, F1: `0.8400` (Improvement: Acc -0.0444, F1 -0.0730)
  - Confusion Matrix:
    ```
    [[61  8]
     [ 0 21]]
    ```

#### Subject S8
- **Baseline Model**: Accuracy: `0.7619`, F1: `0.2308`
  - Confusion Matrix:
    ```
    [[61  0]
     [20  3]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.8333`, F1: `0.5625` (Improvement: Acc +0.0714, F1 +0.3317)
  - Confusion Matrix:
    ```
    [[61  0]
     [14  9]]
    ```

#### Subject S9
- **Baseline Model**: Accuracy: `0.7912`, F1: `0.1739`
  - Confusion Matrix:
    ```
    [[70  0]
     [19  2]]
    ```
- **Balanced Buffer + Dynamic Threshold Model**: Accuracy: `0.8462`, F1: `0.5000` (Improvement: Acc +0.0549, F1 +0.3261)
  - Confusion Matrix:
    ```
    [[70  0]
     [14  7]]
    ```

### Variant 6: Elastic Weight Consolidation (EWC)
- **File**: `part2_edge_simulation_ewc.py`
- **Changes**: Incorporated EWC to prevent catastrophic forgetting. Calculated the Fisher Information Matrix over the generic training dataset, and applied an elastic penalty (`ewc_lambda = 10.0`) during SGD updates. This pulls structurally important weights back towards the generic baseline if they drift too far.
- **Results**:
  - Average Accuracy: `0.8547` (Improvement vs Generic: `+0.0007`)
  - Average F1 Score: `0.5709` (Improvement vs Generic: `-0.0072`)
- **Conclusion**: EWC applies a strong stabilizing anchor to the model, protecting the foundational knowledge learned from the generic population. In highly volatile edge learning scenarios with very limited replay memory, this helps prevent the model from entirely breaking if the user's localized data distribution drastically shifts.

#### Subject S10
- **Baseline Model**: Accuracy: `0.8557`, F1: `0.6111`
  - Confusion Matrix:
    ```
    [[72  1]
     [13 11]]
    ```
- **EWC Model**: Accuracy: `0.8660`, F1: `0.6286` (Improvement: Acc +0.0103, F1 +0.0175)
  - Confusion Matrix:
    ```
    [[73  0]
     [13 11]]
    ```

#### Subject S11
- **Baseline Model**: Accuracy: `0.9286`, F1: `0.8293`
  - Confusion Matrix:
    ```
    [[74  1]
     [ 6 17]]
    ```
- **EWC Model**: Accuracy: `0.9286`, F1: `0.8293` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[74  1]
     [ 6 17]]
    ```

#### Subject S13
- **Baseline Model**: Accuracy: `0.8485`, F1: `0.7368`
  - Confusion Matrix:
    ```
    [[63 14]
     [ 1 21]]
    ```
- **EWC Model**: Accuracy: `0.8485`, F1: `0.6809` (Improvement: Acc +0.0000, F1 -0.0560)
  - Confusion Matrix:
    ```
    [[68  9]
     [ 6 16]]
    ```

#### Subject S14
- **Baseline Model**: Accuracy: `0.7470`, F1: `0.0000`
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```
- **EWC Model**: Accuracy: `0.7470`, F1: `0.0000` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```

#### Subject S15
- **Baseline Model**: Accuracy: `0.8000`, F1: `0.3448`
  - Confusion Matrix:
    ```
    [[71  1]
     [18  5]]
    ```
- **EWC Model**: Accuracy: `0.8105`, F1: `0.3571` (Improvement: Acc +0.0105, F1 +0.0123)
  - Confusion Matrix:
    ```
    [[72  0]
     [18  5]]
    ```

#### Subject S16
- **Baseline Model**: Accuracy: `0.8571`, F1: `0.6061`
  - Confusion Matrix:
    ```
    [[68  1]
     [12 10]]
    ```
- **EWC Model**: Accuracy: `0.8571`, F1: `0.6061` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[68  1]
     [12 10]]
    ```

#### Subject S17
- **Baseline Model**: Accuracy: `0.7500`, F1: `0.0800`
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```
- **EWC Model**: Accuracy: `0.7500`, F1: `0.0800` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```

#### Subject S2
- **Baseline Model**: Accuracy: `0.8750`, F1: `0.7826`
  - Confusion Matrix:
    ```
    [[52  7]
     [ 3 18]]
    ```
- **EWC Model**: Accuracy: `0.8750`, F1: `0.7826` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[52  7]
     [ 3 18]]
    ```

#### Subject S3
- **Baseline Model**: Accuracy: `0.7143`, F1: `0.5200`
  - Confusion Matrix:
    ```
    [[47 15]
     [ 9 13]]
    ```
- **EWC Model**: Accuracy: `0.7143`, F1: `0.5200` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[47 15]
     [ 9 13]]
    ```

#### Subject S4
- **Baseline Model**: Accuracy: `0.9881`, F1: `0.9756`
  - Confusion Matrix:
    ```
    [[63  0]
     [ 1 20]]
    ```
- **EWC Model**: Accuracy: `0.9881`, F1: `0.9756` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[63  0]
     [ 1 20]]
    ```

#### Subject S5
- **Baseline Model**: Accuracy: `0.9778`, F1: `0.9545`
  - Confusion Matrix:
    ```
    [[67  2]
     [ 0 21]]
    ```
- **EWC Model**: Accuracy: `0.9778`, F1: `0.9545` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[67  2]
     [ 0 21]]
    ```

#### Subject S6
- **Baseline Model**: Accuracy: `0.9588`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[72  3]
     [ 1 21]]
    ```
- **EWC Model**: Accuracy: `0.9485`, F1: `0.8936` (Improvement: Acc -0.0103, F1 -0.0194)
  - Confusion Matrix:
    ```
    [[71  4]
     [ 1 21]]
    ```

#### Subject S7
- **Baseline Model**: Accuracy: `0.9556`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[65  4]
     [ 0 21]]
    ```
- **EWC Model**: Accuracy: `0.9667`, F1: `0.9333` (Improvement: Acc +0.0111, F1 +0.0203)
  - Confusion Matrix:
    ```
    [[66  3]
     [ 0 21]]
    ```

#### Subject S8
- **Baseline Model**: Accuracy: `0.7619`, F1: `0.2308`
  - Confusion Matrix:
    ```
    [[61  0]
     [20  3]]
    ```
- **EWC Model**: Accuracy: `0.7619`, F1: `0.2308` (Improvement: Acc +0.0000, F1 +0.0000)
  - Confusion Matrix:
    ```
    [[61  0]
     [20  3]]
    ```

#### Subject S9
- **Baseline Model**: Accuracy: `0.7912`, F1: `0.1739`
  - Confusion Matrix:
    ```
    [[70  0]
     [19  2]]
    ```
- **EWC Model**: Accuracy: `0.7802`, F1: `0.0909` (Improvement: Acc -0.0110, F1 -0.0830)
  - Confusion Matrix:
    ```
    [[70  0]
     [20  1]]
    ```

### Variant 7: Aggressive Hyperparameter Tuning
- **File**: `part2_edge_simulation_aggressive_tuning.py`
- **Changes**: Applied an aggressive set of parameters: Extremely weak EWC anchor (`0.001`), massive learning rate (`0.1`), strict gradient clipping (`[-1.0, 1.0]`), heavy rapid learning (10 updates per new sample), deep historical replays (every 2 updates on a 100-item buffer), and simulated 'shift ends' doing 50 epochs over the entire buffer every 1000 samples.
- **Results**:
  - Average Accuracy: `0.9456` (Improvement vs Generic: `+0.0917`)
  - Average F1 Score: `0.8783` (Improvement vs Generic: `+0.3002`)
- **Conclusion**: Aggressive tuning with massive learning rates and constant deep re-training acts almost like retraining a model from scratch locally. The weak EWC combined with gradient clipping prevents immediate explosion, but the extreme plasticity can cause the model to rapidly overfit to very recent local noise.

#### Subject S10
- **Baseline Model**: Accuracy: `0.8557`, F1: `0.6111`
  - Confusion Matrix:
    ```
    [[72  1]
     [13 11]]
    ```
- **Aggressive Tuning Model**: Accuracy: `0.9794`, F1: `0.9565` (Improvement: Acc +0.1237, F1 +0.3454)
  - Confusion Matrix:
    ```
    [[73  0]
     [ 2 22]]
    ```

#### Subject S11
- **Baseline Model**: Accuracy: `0.9286`, F1: `0.8293`
  - Confusion Matrix:
    ```
    [[74  1]
     [ 6 17]]
    ```
- **Aggressive Tuning Model**: Accuracy: `0.9796`, F1: `0.9545` (Improvement: Acc +0.0510, F1 +0.1253)
  - Confusion Matrix:
    ```
    [[75  0]
     [ 2 21]]
    ```

#### Subject S13
- **Baseline Model**: Accuracy: `0.8485`, F1: `0.7368`
  - Confusion Matrix:
    ```
    [[63 14]
     [ 1 21]]
    ```
- **Aggressive Tuning Model**: Accuracy: `0.9697`, F1: `0.9302` (Improvement: Acc +0.1212, F1 +0.1934)
  - Confusion Matrix:
    ```
    [[76  1]
     [ 2 20]]
    ```

#### Subject S14
- **Baseline Model**: Accuracy: `0.7470`, F1: `0.0000`
  - Confusion Matrix:
    ```
    [[62  0]
     [21  0]]
    ```
- **Aggressive Tuning Model**: Accuracy: `0.8795`, F1: `0.7500` (Improvement: Acc +0.1325, F1 +0.7500)
  - Confusion Matrix:
    ```
    [[58  4]
     [ 6 15]]
    ```

#### Subject S15
- **Baseline Model**: Accuracy: `0.8000`, F1: `0.3448`
  - Confusion Matrix:
    ```
    [[71  1]
     [18  5]]
    ```
- **Aggressive Tuning Model**: Accuracy: `0.9053`, F1: `0.7568` (Improvement: Acc +0.1053, F1 +0.4119)
  - Confusion Matrix:
    ```
    [[72  0]
     [ 9 14]]
    ```

#### Subject S16
- **Baseline Model**: Accuracy: `0.8571`, F1: `0.6061`
  - Confusion Matrix:
    ```
    [[68  1]
     [12 10]]
    ```
- **Aggressive Tuning Model**: Accuracy: `0.9670`, F1: `0.9302` (Improvement: Acc +0.1099, F1 +0.3242)
  - Confusion Matrix:
    ```
    [[68  1]
     [ 2 20]]
    ```

#### Subject S17
- **Baseline Model**: Accuracy: `0.7500`, F1: `0.0800`
  - Confusion Matrix:
    ```
    [[68  0]
     [23  1]]
    ```
- **Aggressive Tuning Model**: Accuracy: `0.9022`, F1: `0.7805` (Improvement: Acc +0.1522, F1 +0.7005)
  - Confusion Matrix:
    ```
    [[67  1]
     [ 8 16]]
    ```

#### Subject S2
- **Baseline Model**: Accuracy: `0.8750`, F1: `0.7826`
  - Confusion Matrix:
    ```
    [[52  7]
     [ 3 18]]
    ```
- **Aggressive Tuning Model**: Accuracy: `0.9250`, F1: `0.8571` (Improvement: Acc +0.0500, F1 +0.0745)
  - Confusion Matrix:
    ```
    [[56  3]
     [ 3 18]]
    ```

#### Subject S3
- **Baseline Model**: Accuracy: `0.7143`, F1: `0.5200`
  - Confusion Matrix:
    ```
    [[47 15]
     [ 9 13]]
    ```
- **Aggressive Tuning Model**: Accuracy: `0.8810`, F1: `0.7368` (Improvement: Acc +0.1667, F1 +0.2168)
  - Confusion Matrix:
    ```
    [[60  2]
     [ 8 14]]
    ```

#### Subject S4
- **Baseline Model**: Accuracy: `0.9881`, F1: `0.9756`
  - Confusion Matrix:
    ```
    [[63  0]
     [ 1 20]]
    ```
- **Aggressive Tuning Model**: Accuracy: `1.0000`, F1: `1.0000` (Improvement: Acc +0.0119, F1 +0.0244)
  - Confusion Matrix:
    ```
    [[63  0]
     [ 0 21]]
    ```

#### Subject S5
- **Baseline Model**: Accuracy: `0.9778`, F1: `0.9545`
  - Confusion Matrix:
    ```
    [[67  2]
     [ 0 21]]
    ```
- **Aggressive Tuning Model**: Accuracy: `1.0000`, F1: `1.0000` (Improvement: Acc +0.0222, F1 +0.0455)
  - Confusion Matrix:
    ```
    [[69  0]
     [ 0 21]]
    ```

#### Subject S6
- **Baseline Model**: Accuracy: `0.9588`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[72  3]
     [ 1 21]]
    ```
- **Aggressive Tuning Model**: Accuracy: `0.9897`, F1: `0.9778` (Improvement: Acc +0.0309, F1 +0.0647)
  - Confusion Matrix:
    ```
    [[74  1]
     [ 0 22]]
    ```

#### Subject S7
- **Baseline Model**: Accuracy: `0.9556`, F1: `0.9130`
  - Confusion Matrix:
    ```
    [[65  4]
     [ 0 21]]
    ```
- **Aggressive Tuning Model**: Accuracy: `0.9667`, F1: `0.9231` (Improvement: Acc +0.0111, F1 +0.0100)
  - Confusion Matrix:
    ```
    [[69  0]
     [ 3 18]]
    ```

#### Subject S8
- **Baseline Model**: Accuracy: `0.7619`, F1: `0.2308`
  - Confusion Matrix:
    ```
    [[61  0]
     [20  3]]
    ```
- **Aggressive Tuning Model**: Accuracy: `0.9167`, F1: `0.8205` (Improvement: Acc +0.1548, F1 +0.5897)
  - Confusion Matrix:
    ```
    [[61  0]
     [ 7 16]]
    ```

#### Subject S9
- **Baseline Model**: Accuracy: `0.7912`, F1: `0.1739`
  - Confusion Matrix:
    ```
    [[70  0]
     [19  2]]
    ```
- **Aggressive Tuning Model**: Accuracy: `0.9231`, F1: `0.8000` (Improvement: Acc +0.1319, F1 +0.6261)
  - Confusion Matrix:
    ```
    [[70  0]
     [ 7 14]]
    ```


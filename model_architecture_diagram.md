# Model Architecture: Two-Stage Enhanced System

This diagram illustrates the data flow and architecture implemented in `improved_bank_marketing.py`.

```mermaid
graph TD
    subgraph Data_Preparation ["Stage 1: Data Preparation"]
        Raw[Raw Data] --> Preprocessing[Preprocessing & Feature Engineering]
        Preprocessing --> Encoded[Encoded Features X]
        Preprocessing --> DurationTarget[Duration Target<br/>(Median Split: Short/Long)]
    end

    subgraph Duration_Model ["Stage 2: Duration Classifier"]
        Encoded --> DurModel[LightGBM Classifier]
        DurationTarget --> DurModel
        DurModel --> DurProbs[Predicted Probabilities<br/>(P_Short, P_Long)]
    end

    subgraph Feature_Enhancement ["Stage 3: Feature Enhancement"]
        Encoded --> EnhancedX[Enhanced Feature Set]
        DurProbs --> EnhancedX
        Note[Add P_Short & P_Long<br/>as new features] -.-> EnhancedX
    end

    subgraph Outcome_Model ["Stage 4: Outcome Prediction"]
        EnhancedX --> Pipeline[Imbalanced Pipeline]
        Pipeline --> SMOTE[SMOTE Oversampling]
        SMOTE --> OutcomeLGBM[LightGBM Classifier<br/>(Predict Success)]
        OutcomeLGBM --> SuccessProbs[Success Probability<br/>(P_Success)]
    end

    subgraph Strategy ["Stage 5: Decision Support"]
        DurProbs --> Calculator[Efficiency Calculator]
        SuccessProbs --> Calculator
        Calculator --> Efficiency[Efficiency Score = P_Success / E_Duration]
        Efficiency --> Schedule[Prioritized Call Schedule]
    end

    style Data_Preparation fill:#f9f,stroke:#333
    style Duration_Model fill:#bbf,stroke:#333
    style Feature_Enhancement fill:#bfb,stroke:#333
    style Outcome_Model fill:#fbf,stroke:#333
    style Strategy fill:#ff9,stroke:#333
```

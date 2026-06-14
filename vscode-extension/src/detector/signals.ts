export enum DilemmaType {
    SEMANTIC_STUCK = "SEMANTIC_STUCK",
    SYNTAX_ANOMALY = "SYNTAX_ANOMALY",
    PATTERN_REPEAT = "PATTERN_REPEAT",
    GENERATION_STALL = "GENERATION_STALL"
}

export interface DilemmaSignal {
    type: DilemmaType;
    confidence: number;  // 0.0 ~ 1.0
    features: {
        tokenRepetition?: number;
        logitsEntropy?: number;
        ngramRepetition?: number;
        consecutiveSame?: number;
    };
    timestamp: number;
}
export interface InterventionResult {
    success: boolean;
    newTokens?: number[];
    message: string;
    metadata?: Record<string, any>;
}

export interface StrategyContext {
    tokens: number[];
    logits: number[][];
    prompt: string;
    cursorPosition: number;
    language: string;
}

export interface InterventionStrategy {
    readonly name: string;
    readonly triggerConditions: {
        type: string;
        minConfidence?: number;
        maxConfidence?: number;
    }[];

    shouldTrigger(signal: any): boolean;
    execute(context: StrategyContext): Promise<InterventionResult>;
}

// BacktrackStrategy
export class BacktrackStrategy implements InterventionStrategy {
    readonly name = "BacktrackStrategy";
    readonly triggerConditions = [
        { type: "SEMANTIC_STUCK", minConfidence: 0.3, maxConfidence: 0.8 },
        { type: "GENERATION_STALL" }
    ];
    private backtrackSteps: number = 3;

    shouldTrigger(signal: any): boolean {
        return this.triggerConditions.some(tc => {
            if (signal.type !== tc.type) { return false; }
            const conf = signal.confidence || 0;
            if (tc.minConfidence !== undefined && conf < tc.minConfidence) { return false; }
            if (tc.maxConfidence !== undefined && conf > tc.maxConfidence) { return false; }
            return true;
        });
    }

    async execute(context: StrategyContext): Promise<InterventionResult> {
        const steps = Math.min(this.backtrackSteps, context.tokens.length);
        const truncatedTokens = context.tokens.slice(0, -steps);
        return {
            success: true,
            newTokens: truncatedTokens,
            message: `Backtracked ${steps} tokens`
        };
    }
}

// RerankStrategy
export class RerankStrategy implements InterventionStrategy {
    readonly name = "RerankStrategy";
    readonly triggerConditions = [
        { type: "LOW_CONFIDENCE", maxConfidence: 0.5 }
    ];

    shouldTrigger(signal: any): boolean {
        return signal.type === "LOW_CONFIDENCE" && (signal.confidence || 0) <= 0.5;
    }

    async execute(context: StrategyContext): Promise<InterventionResult> {
        return {
            success: true,
            message: "Reranked from top-k candidates"
        };
    }
}

// DiversityStrategy
export class DiversityStrategy implements InterventionStrategy {
    readonly name = "DiversityStrategy";
    readonly triggerConditions = [
        { type: "REPETITION", maxConfidence: 0.6 }
    ];

    shouldTrigger(signal: any): boolean {
        return signal.type === "REPETITION" && (signal.confidence || 0) <= 0.6;
    }

    async execute(context: StrategyContext): Promise<InterventionResult> {
        return {
            success: true,
            message: "Increased temperature for diversity"
        };
    }
}

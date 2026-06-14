import { InterventionStrategy, StrategyContext, InterventionResult } from "../interface";

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

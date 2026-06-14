import { InterventionStrategy, StrategyContext, InterventionResult } from "./interface";

export class StrategyRouter {
    private strategies: InterventionStrategy[] = [];
    private defaultStrategy?: InterventionStrategy;
    
    register(strategy: InterventionStrategy): void {
        this.strategies.push(strategy);
    }
    
    unregister(strategyName: string): void {
        this.strategies = this.strategies.filter(s => s.name !== strategyName);
    }
    
    setDefaultStrategy(strategy: InterventionStrategy): void {
        this.defaultStrategy = strategy;
    }
    
    async route(signal: any, context: StrategyContext): Promise<InterventionResult> {
        for (const strategy of this.strategies) {
            if (strategy.shouldTrigger(signal)) {
                try {
                    const result = await strategy.execute(context);
                    if (result.success) {
                        return result;
                    }
                } catch (error) {
                    console.error(`Strategy ${strategy.name} failed:`, error);
                }
            }
        }
        
        if (this.defaultStrategy) {
            return this.defaultStrategy.execute(context);
        }
        
        return {
            success: false,
            message: "No applicable strategy found"
        };
    }
    
    listStrategies(): string[] {
        return this.strategies.map(s => s.name);
    }
}

import { DilemmaSignal, DilemmaType } from "./signals";
import { DEFAULT_THRESHOLDS } from "./thresholds";

export class DilemmaDetector {
    private thresholds: typeof DEFAULT_THRESHOLDS;
    private cache: Map<string, DilemmaSignal>;
    private cacheMaxSize: number = 100;

    constructor(thresholds?: Partial<typeof DEFAULT_THRESHOLDS>) {
        this.thresholds = { ...DEFAULT_THRESHOLDS, ...thresholds };
        this.cache = new Map();
    }

    public detect(tokens: number[], logits: number[][]): DilemmaSignal {
        const cacheKey = this.generateCacheKey(tokens, logits);
        if (this.cache.has(cacheKey)) {
            return { ...this.cache.get(cacheKey)! };
        }

        const consecutive = this.detectConsecutiveSame(tokens);
        if (consecutive > this.thresholds.consecutiveSame) {
            const result = this.buildResult(DilemmaType.GENERATION_STALL, 0, 0, 0, consecutive);
            this.addToCache(cacheKey, result);
            return result;
        }

        const tokenRep = this.computeTokenRepetition(tokens);
        if (tokenRep > this.thresholds.tokenRepetition) {
            const result = this.buildResult(DilemmaType.SEMANTIC_STUCK, tokenRep, 0, 0, consecutive);
            this.addToCache(cacheKey, result);
            return result;
        }

        const entropy = logits.length > 0 ? this.computeLogitsEntropy(logits) : 0;
        if (entropy > this.thresholds.logitsEntropy) {
            const result = this.buildResult(DilemmaType.SYNTAX_ANOMALY, tokenRep, entropy, 0, consecutive);
            this.addToCache(cacheKey, result);
            return result;
        }

        const ngramRep = this.computeNgramRepetition(tokens);
        const type = ngramRep > this.thresholds.ngramRepetition ? DilemmaType.PATTERN_REPEAT : DilemmaType.SEMANTIC_STUCK;
        
        const result = this.buildResult(type, tokenRep, entropy, ngramRep, consecutive);
        this.addToCache(cacheKey, result);
        return result;
    }

    private buildResult(type: DilemmaType, tokenRep: number, entropy: number, ngramRep: number, consecutive: number): DilemmaSignal {
        const confidence = this.computeConfidence(type, tokenRep, entropy, ngramRep, consecutive);
        return {
            type,
            confidence,
            features: { tokenRepetition: tokenRep, logitsEntropy: entropy, ngramRepetition: ngramRep, consecutiveSame: consecutive },
            timestamp: Date.now()
        };
    }

    private generateCacheKey(tokens: number[], logits: number[][]): string {
        const tokenHash = tokens.length < 100 ? tokens.join(',') : `${tokens.length}:${tokens[0]}:${tokens[tokens.length-1]}`;
        const logitsHash = logits.length < 10 ? JSON.stringify(logits) : `${logits.length}`;
        return `${tokenHash}|${logitsHash}`;
    }

    private addToCache(key: string, value: DilemmaSignal): void {
        if (this.cache.size >= this.cacheMaxSize) {
            const firstKey = this.cache.keys().next().value as string;
            this.cache.delete(firstKey);
        }
        this.cache.set(key, value);
    }

    private computeTokenRepetition(tokens: number[]): number {
        const window = 5;
        const len = tokens.length;
        if (len <= window) return 0;
        
        let repeats = 0;
        const maxIdx = len - window;
        for (let i = 0; i < maxIdx; i++) {
            if (tokens[i] === tokens[i + 1]) repeats++;
        }
        return repeats / maxIdx;
    }

    private computeLogitsEntropy(logits: number[][]): number {
        const len = logits.length;
        if (len === 0) return 0;

        let totalEntropy = 0;
        for (let i = 0; i < len; i++) {
            totalEntropy += this.computeSingleEntropy(logits[i]);
        }
        return totalEntropy / len;
    }

    private computeSingleEntropy(logit: number[]): number {
        const len = logit.length;
        if (len === 0) return 0;

        let maxLogit = logit[0];
        for (let i = 1; i < len; i++) {
            if (logit[i] > maxLogit) maxLogit = logit[i];
        }

        let sumExp = 0;
        for (let i = 0; i < len; i++) {
            sumExp += Math.exp(logit[i] - maxLogit);
        }

        let entropy = 0;
        const invSumExp = 1 / sumExp;
        for (let i = 0; i < len; i++) {
            const p = Math.exp(logit[i] - maxLogit) * invSumExp;
            if (p > 0) entropy -= p * Math.log2(p);
        }
        return entropy;
    }

    private computeNgramRepetition(tokens: number[], n: number = 3): number {
        const len = tokens.length;
        if (len <= n) return 0;

        const ngramCount = new Map<number, number>();
        const maxIdx = len - n;
        
        for (let i = 0; i < maxIdx; i++) {
            const hash = this.hashNgram(tokens, i, n);
            ngramCount.set(hash, (ngramCount.get(hash) || 0) + 1);
        }

        let repeats = 0;
        for (const count of ngramCount.values()) {
            if (count > 1) repeats++;
        }
        return repeats / maxIdx;
    }

    private hashNgram(tokens: number[], start: number, n: number): number {
        let hash = 0;
        for (let i = 0; i < n; i++) {
            hash = (hash * 31) + tokens[start + i];
        }
        return hash;
    }

    private detectConsecutiveSame(tokens: number[]): number {
        const len = tokens.length;
        if (len === 0) return 0;

        let maxConsecutive = 1;
        let current = 1;
        for (let i = 1; i < len; i++) {
            if (tokens[i] === tokens[i - 1]) {
                if (++current > maxConsecutive) {
                    maxConsecutive = current;
                }
            } else {
                current = 1;
            }
        }
        return maxConsecutive;
    }

    private computeConfidence(type: DilemmaType, tokenRep: number, entropy: number, ngramRep: number, consecutive: number): number {
        switch (type) {
            case DilemmaType.GENERATION_STALL:
                return Math.min(consecutive / (this.thresholds.consecutiveSame * 2), 1);
            case DilemmaType.SEMANTIC_STUCK:
                return Math.min(tokenRep / (this.thresholds.tokenRepetition * 2), 1);
            case DilemmaType.SYNTAX_ANOMALY:
                return Math.min(entropy / (this.thresholds.logitsEntropy * 2), 1);
            case DilemmaType.PATTERN_REPEAT:
                return Math.min(ngramRep / (this.thresholds.ngramRepetition * 2), 1);
            default:
                return 0;
        }
    }

    public resetCache(): void {
        this.cache.clear();
    }

    public setCacheSize(size: number): void {
        this.cacheMaxSize = size;
    }
}
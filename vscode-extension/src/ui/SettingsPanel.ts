import * as vscode from "vscode";

export interface MetacogSettings {
    enabled: boolean;
    llmProvider: "openai" | "ollama" | "claude";
    apiKey?: string;
    baseURL?: string;
    model?: string;
    thresholds: {
        tokenRepetition: number;
        logitsEntropy: number;
        ngramRepetition: number;
        consecutiveSame: number;
    };
}

export const DEFAULT_SETTINGS: MetacogSettings = {
    enabled: true,
    llmProvider: "ollama",
    baseURL: "http://localhost:11434",
    model: "codellama",
    thresholds: {
        tokenRepetition: 0.3,
        logitsEntropy: 2.5,
        ngramRepetition: 0.4,
        consecutiveSame: 3
    }
};

export class SettingsManager {
    private config: vscode.WorkspaceConfiguration;
    
    constructor() {
        this.config = vscode.workspace.getConfiguration("metacog-x");
    }
    
    getSettings(): MetacogSettings {
        return {
            enabled: this.config.get("enabled", DEFAULT_SETTINGS.enabled),
            llmProvider: this.config.get("llmProvider", DEFAULT_SETTINGS.llmProvider),
            apiKey: this.config.get("apiKey", DEFAULT_SETTINGS.apiKey),
            baseURL: this.config.get("baseURL", DEFAULT_SETTINGS.baseURL),
            model: this.config.get("model", DEFAULT_SETTINGS.model),
            thresholds: {
                tokenRepetition: this.config.get("thresholds.tokenRepetition", DEFAULT_SETTINGS.thresholds.tokenRepetition),
                logitsEntropy: this.config.get("thresholds.logitsEntropy", DEFAULT_SETTINGS.thresholds.logitsEntropy),
                ngramRepetition: this.config.get("thresholds.ngramRepetition", DEFAULT_SETTINGS.thresholds.ngramRepetition),
                consecutiveSame: this.config.get("thresholds.consecutiveSame", DEFAULT_SETTINGS.thresholds.consecutiveSame)
            }
        };
    }
    
    updateSettings(settings: Partial<MetacogSettings>): void {
        for (const [key, value] of Object.entries(settings)) {
            if (key === "thresholds") {
                for (const [tkey, tvalue] of Object.entries(value as any)) {
                    this.config.set(`thresholds.${tkey}`, tvalue);
                }
            } else {
                this.config.set(key, value);
            }
        }
    }
}

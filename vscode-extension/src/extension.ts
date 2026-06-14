import * as vscode from "vscode";
import { DilemmaDetector } from "./detector/DilemmaDetector";
import { StrategyRouter } from "./strategy/StrategyRouter";
import { BacktrackStrategy, RerankStrategy, DiversityStrategy } from "./strategy/interface";
import { OpenAIBackend } from "./backend/OpenAIBackend";
import { OllamaBackend } from "./backend/OllamaBackend";
import { StatusBarManager, MetacogMode } from "./ui/StatusBar";
import { SettingsManager } from "./ui/SettingsPanel";

// 全局状态
let detector: DilemmaDetector;
let router: StrategyRouter;
let statusBar: StatusBarManager;
let settings: SettingsManager;
let llmBackend: any;

// 插件激活
export function activate(context: vscode.ExtensionContext) {
    console.log("MetaCog-X extension activating...");

    // 初始化组件
    settings = new SettingsManager();
    const config = settings.getSettings();

    detector = new DilemmaDetector(config.thresholds);
    router = new StrategyRouter();

    // 注册策略
    router.register(new BacktrackStrategy());
    router.register(new RerankStrategy());
    router.register(new DiversityStrategy());

    // 初始化 LLM 后端
    if (config.llmProvider === "openai") {
        llmBackend = new OpenAIBackend(config.apiKey || "", config.model);
    } else {
        llmBackend = new OllamaBackend(config.baseURL, config.model);
    }

    // 初始化 UI
    statusBar = new StatusBarManager();

    // 注册命令
    const commands = [
        vscode.commands.registerCommand("metacog-x.showStatus", showStatus),
        vscode.commands.registerCommand("metacog-x.toggle", toggleEnabled),
        vscode.commands.registerCommand("metacog-x.showSettings", showSettings)
    ];

    commands.forEach(cmd => context.subscriptions.push(cmd));

    // 注册代码补全事件
    vscode.languages.registerInlineCompletionItemProvider(
        { pattern: "**" },
        new MetaCogCompletionProvider()
    );

    console.log("MetaCog-X extension activated!");
}

// 代码补全 Provider
class MetaCogCompletionProvider implements vscode.InlineCompletionItemProvider {
    private pendingRequests: Map<string, vscode.CancellationTokenSource> = new Map();

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[]> {
        // 1. 获取设置
        const config = settings.getSettings();
        if (!config.enabled) { return []; }

        // 2. 获取上下文
        const line = document.lineAt(position).text.substring(0, position.character);
        const prompt = this.buildPrompt(line);

        // 3. 发送请求到 LLM
        try {
            const response = await llmBackend.complete({
                prompt,
                maxTokens: 50,
                temperature: 0.7
            });

            // 4. 解析生成的代码
            const completion = response.text.trim();
            if (!completion) { return []; }

            // 5. 困境检测
            const tokens = this.tokenize(completion);
            const logits: number[][] = []; // 简化处理
            const signal = detector.detect(tokens, logits);

            // 6. 根据信号更新 UI
            if (signal.confidence > 0.5) {
                statusBar.setMode(MetacogMode.Metacog);

                // 7. 如果检测到困境，触发干预
                const result = await router.route(signal, {
                    tokens,
                    logits,
                    prompt,
                    cursorPosition: position.character,
                    language: document.languageId
                });

                if (result.success && result.newTokens) {
                    // 使用干预后的结果
                    statusBar.setMode(MetacogMode.Stuck);
                }
            } else {
                statusBar.setMode(MetacogMode.Normal);
            }

            // 8. 返回补全项
            return [{
                insertText: completion,
                range: new vscode.Range(position, position)
            }];

        } catch (error) {
            console.error("Completion error:", error);
            statusBar.setMode(MetacogMode.Normal);
            return [];
        }
    }

    private buildPrompt(line: string): string {
        // 简单的提示构建
        return `Continue the following code:\n${line}`;
    }

    private tokenize(text: string): number[] {
        // 简单的token化
        return Array.from(text).map(c => c.charCodeAt(0));
    }
}

function showStatus() {
    const config = settings.getSettings();
    vscode.window.showInformationMessage(
        `MetaCog-X Status: ${config.enabled ? "Enabled" : "Disabled"} | Provider: ${config.llmProvider}`
    );
}

function toggleEnabled() {
    const config = settings.getSettings();
    settings.updateSettings({ enabled: !config.enabled });
    vscode.window.showInformationMessage(
        `MetaCog-X ${config.enabled ? "Disabled" : "Enabled"}`
    );
}

function showSettings() {
    vscode.commands.executeCommand("workbench.action.openSettings", "metacog-x");
}

// 插件停用
export function deactivate() {
    if (statusBar) {
        statusBar.dispose();
    }
}

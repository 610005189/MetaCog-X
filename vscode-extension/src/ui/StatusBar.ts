import * as vscode from "vscode";

export enum MetacogMode {
    Normal = "Normal",
    Metacog = "Metacog",
    Stuck = "Stuck"
}

export class StatusBarManager {
    private statusBar: vscode.StatusBarItem;
    private currentMode: MetacogMode = MetacogMode.Normal;
    
    constructor() {
        this.statusBar = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        this.statusBar.command = "metacog-x.showStatus";
        this.updateDisplay();
    }
    
    setMode(mode: MetacogMode): void {
        this.currentMode = mode;
        this.updateDisplay();
    }
    
    private updateDisplay(): void {
        const icons: Record<MetacogMode, string> = {
            [MetacogMode.Normal]: "$(check)",
            [MetacogMode.Metacog]: "$(gear)",
            [MetacogMode.Stuck]: "$(warning)"
        };
        
        const colors: Record<MetacogMode, string> = {
            [MetacogMode.Normal]: "normal",
            [MetacogMode.Metacog]: "突出",
            [MetacogMode.Stuck]: "notifications"
        };
        
        this.statusBar.text = `${icons[this.currentMode]} MetaCog-X: ${this.currentMode}`;
        this.statusBar.color = colors[this.currentMode];
        this.statusBar.show();
    }
    
    dispose(): void {
        this.statusBar.dispose();
    }
}

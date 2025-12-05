"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const binaryViewer_1 = require("./binaryViewer");
const treeProviders_1 = require("./treeProviders");
const parser_1 = require("./parser");
let parser;
let overviewProvider;
let blocksProvider;
let provenanceProvider;
function activate(context) {
    console.log('MAIF Explorer extension is now active');
    // Initialize parser
    parser = new parser_1.MAIFParser();
    // Register custom editor for .maif files
    const binaryViewerProvider = new binaryViewer_1.MAIFBinaryViewerProvider(context);
    context.subscriptions.push(vscode.window.registerCustomEditorProvider('maif.binaryViewer', binaryViewerProvider, {
        webviewOptions: { retainContextWhenHidden: true },
        supportsMultipleEditorsPerDocument: false
    }));
    // Register tree view providers
    overviewProvider = new treeProviders_1.MAIFOverviewProvider();
    blocksProvider = new treeProviders_1.MAIFBlocksProvider();
    provenanceProvider = new treeProviders_1.MAIFProvenanceProvider();
    context.subscriptions.push(vscode.window.registerTreeDataProvider('maifOverview', overviewProvider), vscode.window.registerTreeDataProvider('maifBlocks', blocksProvider), vscode.window.registerTreeDataProvider('maifProvenance', provenanceProvider));
    // Register commands
    context.subscriptions.push(vscode.commands.registerCommand('maif.openExplorer', async (uri) => {
        if (!uri) {
            const uris = await vscode.window.showOpenDialog({
                canSelectMany: false,
                filters: {
                    'MAIF Files': ['maif', 'json']
                }
            });
            if (uris && uris.length > 0) {
                uri = uris[0];
            }
        }
        if (uri) {
            await openMAIFFile(uri);
        }
    }), vscode.commands.registerCommand('maif.showProvenance', async () => {
        const editor = vscode.window.activeTextEditor;
        if (editor && parser?.manifest) {
            const provenance = parser.manifest.signature_metadata?.provenance_chain || [];
            const panel = vscode.window.createWebviewPanel('maifProvenance', 'MAIF Provenance Chain', vscode.ViewColumn.Beside, { enableScripts: true });
            panel.webview.html = getProvenanceHtml(provenance);
        }
        else {
            vscode.window.showWarningMessage('No MAIF manifest loaded');
        }
    }), vscode.commands.registerCommand('maif.verifySignature', async () => {
        if (parser?.manifest?.signature) {
            vscode.window.showInformationMessage(`Signature present from: ${parser.manifest.signature_metadata?.signer_id || 'Unknown'}`);
        }
        else {
            vscode.window.showWarningMessage('No signature found in manifest');
        }
    }), vscode.commands.registerCommand('maif.showHexView', async () => {
        if (parser?.binaryData) {
            const panel = vscode.window.createWebviewPanel('maifHexView', 'MAIF Hex View', vscode.ViewColumn.Beside, { enableScripts: true });
            panel.webview.html = getHexViewHtml(parser.binaryData);
        }
        else {
            vscode.window.showWarningMessage('No binary data loaded');
        }
    }), vscode.commands.registerCommand('maif.selectBlock', async (blockIndex) => {
        if (parser?.blocks && blockIndex >= 0 && blockIndex < parser.blocks.length) {
            const block = parser.blocks[blockIndex];
            const panel = vscode.window.createWebviewPanel('maifBlockDetail', `Block: ${block.type}`, vscode.ViewColumn.Beside, { enableScripts: true });
            panel.webview.html = getBlockDetailHtml(block);
        }
    }));
    // Watch for document changes
    context.subscriptions.push(vscode.workspace.onDidOpenTextDocument(async (doc) => {
        if (doc.fileName.endsWith('_manifest.json') || doc.fileName.endsWith('-manifest.json')) {
            await loadManifest(doc.uri);
        }
    }));
}
async function openMAIFFile(uri) {
    try {
        if (uri.fsPath.endsWith('.maif')) {
            // Load binary first - it will auto-detect format
            await loadBinary(uri);
            // If it's legacy format (not secure), try to find accompanying manifest
            if (parser?.format !== 'secure') {
                const manifestPath = uri.fsPath.replace('.maif', '_manifest.json');
                const manifestUri = vscode.Uri.file(manifestPath);
                try {
                    await vscode.workspace.fs.stat(manifestUri);
                    await loadManifest(manifestUri);
                }
                catch {
                    // Manifest not found, continue with binary-only view
                }
            }
            await vscode.commands.executeCommand('vscode.openWith', uri, 'maif.binaryViewer');
            // Show format info
            const formatMsg = parser?.format === 'secure'
                ? ' Secure MAIF format (self-contained with embedded security)'
                : '📁 Legacy MAIF format';
            vscode.window.setStatusBarMessage(formatMsg, 5000);
        }
        else if (uri.fsPath.endsWith('.json')) {
            await loadManifest(uri);
            await vscode.window.showTextDocument(uri);
        }
        refreshTreeViews();
    }
    catch (error) {
        vscode.window.showErrorMessage(`Failed to open MAIF file: ${error}`);
    }
}
async function loadManifest(uri) {
    const content = await vscode.workspace.fs.readFile(uri);
    const text = new TextDecoder().decode(content);
    parser?.loadManifestFromString(text);
}
async function loadBinary(uri) {
    const content = await vscode.workspace.fs.readFile(uri);
    parser?.loadBinaryFromBuffer(content);
}
function refreshTreeViews() {
    if (parser?.manifest) {
        overviewProvider?.setManifest(parser.manifest, parser.fileInfo || undefined, parser.securityInfo || undefined);
        blocksProvider?.setBlocks(parser.blocks);
        provenanceProvider?.setProvenance(parser.provenance || []);
    }
}
function getProvenanceHtml(provenance) {
    const entries = provenance.map((entry, i) => `
        <div class="entry ${entry.action === 'genesis' ? 'genesis' : ''}">
            <div class="marker"></div>
            <div class="content">
                <div class="action">${formatAction(entry.action)}</div>
                <div class="time">${formatTimestamp(entry.timestamp)}</div>
                <div class="details">
                    <div><strong>Agent:</strong> ${entry.agent_id}</div>
                    <div><strong>DID:</strong> ${entry.agent_did || 'N/A'}</div>
                    <div><strong>Entry Hash:</strong> <code>${truncate(entry.entry_hash)}</code></div>
                    <div><strong>Block Hash:</strong> <code>${truncate(entry.block_hash)}</code></div>
                </div>
                ${entry.signature ? `<div class="signature"><strong>Signature:</strong> <code>${truncate(entry.signature, 50)}</code></div>` : ''}
            </div>
        </div>
    `).join('');
    return `<!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: var(--vscode-font-family); padding: 20px; color: var(--vscode-foreground); }
            h1 { font-size: 18px; margin-bottom: 20px; }
            .timeline { position: relative; padding-left: 30px; }
            .timeline::before { content: ''; position: absolute; left: 10px; top: 0; bottom: 0; width: 2px; background: var(--vscode-textSeparator-foreground); }
            .entry { position: relative; margin-bottom: 20px; padding: 15px; background: var(--vscode-editor-inactiveSelectionBackground); border-radius: 8px; }
            .entry.genesis { border-left: 3px solid var(--vscode-charts-green); }
            .marker { position: absolute; left: -26px; top: 20px; width: 12px; height: 12px; border-radius: 50%; background: var(--vscode-button-background); }
            .genesis .marker { background: var(--vscode-charts-green); }
            .action { font-weight: bold; color: var(--vscode-textLink-foreground); margin-bottom: 5px; }
            .time { font-size: 12px; color: var(--vscode-descriptionForeground); margin-bottom: 10px; }
            .details { font-size: 13px; line-height: 1.6; }
            .signature { margin-top: 10px; font-size: 12px; padding-top: 10px; border-top: 1px solid var(--vscode-textSeparator-foreground); }
            code { background: var(--vscode-textCodeBlock-background); padding: 2px 6px; border-radius: 3px; font-size: 11px; }
        </style>
    </head>
    <body>
        <h1> Provenance Chain (${provenance.length} entries)</h1>
        <div class="timeline">${entries}</div>
    </body>
    </html>`;
}
function getHexViewHtml(data) {
    const lines = [];
    for (let i = 0; i < Math.min(data.length, 2048); i += 16) {
        const offset = i.toString(16).padStart(8, '0');
        const bytes = Array.from(data.slice(i, Math.min(i + 16, data.length)));
        const hex = bytes.map(b => b.toString(16).padStart(2, '0')).join(' ').padEnd(47, ' ');
        const ascii = bytes.map(b => (b >= 32 && b < 127) ? String.fromCharCode(b) : '.').join('');
        lines.push(`<div class="line"><span class="offset">${offset}</span><span class="hex">${hex}</span><span class="ascii">${ascii}</span></div>`);
    }
    return `<!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: var(--vscode-editor-font-family); font-size: 13px; padding: 20px; color: var(--vscode-foreground); }
            h1 { font-size: 16px; margin-bottom: 15px; }
            .info { font-size: 12px; color: var(--vscode-descriptionForeground); margin-bottom: 20px; }
            .hex-view { background: var(--vscode-editor-background); padding: 15px; border-radius: 8px; overflow: auto; }
            .line { display: flex; gap: 20px; line-height: 1.8; }
            .offset { color: var(--vscode-descriptionForeground); min-width: 80px; }
            .hex { color: var(--vscode-textLink-foreground); min-width: 360px; }
            .ascii { color: var(--vscode-foreground); }
        </style>
    </head>
    <body>
        <h1>🔢 Hex View</h1>
        <div class="info">${data.length.toLocaleString()} bytes total (showing first 2KB)</div>
        <div class="hex-view">${lines.join('')}</div>
    </body>
    </html>`;
}
function getBlockDetailHtml(block) {
    const statusBadges = [];
    if (block.isSigned)
        statusBadges.push('<span class="status-badge signed"> Signed</span>');
    if (block.isImmutable)
        statusBadges.push('<span class="status-badge immutable"> Immutable</span>');
    if (block.isTampered)
        statusBadges.push('<span class="status-badge tampered"> TAMPERED</span>');
    return `<!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: var(--vscode-font-family); padding: 20px; color: var(--vscode-foreground); }
            h1 { font-size: 18px; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
            .badge { background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); padding: 4px 8px; border-radius: 4px; font-size: 12px; }
            .status-badges { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
            .status-badge { padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 500; }
            .status-badge.signed { background: rgba(16, 185, 129, 0.15); color: #10b981; }
            .status-badge.immutable { background: rgba(99, 102, 241, 0.15); color: #6366f1; }
            .status-badge.tampered { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
            .section { margin-bottom: 20px; }
            .section-title { font-size: 12px; text-transform: uppercase; color: var(--vscode-descriptionForeground); margin-bottom: 10px; }
            .row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--vscode-textSeparator-foreground); }
            .label { color: var(--vscode-descriptionForeground); }
            .value { font-family: var(--vscode-editor-font-family); }
            .hash { font-family: var(--vscode-editor-font-family); font-size: 11px; background: var(--vscode-textCodeBlock-background); padding: 10px; border-radius: 6px; word-break: break-all; }
            pre { background: var(--vscode-textCodeBlock-background); padding: 15px; border-radius: 6px; overflow: auto; font-size: 12px; }
        </style>
    </head>
    <body>
        <h1><span class="badge">${block.type}</span> Block Details</h1>
        
        ${statusBadges.length > 0 ? `<div class="status-badges">${statusBadges.join('')}</div>` : ''}
        
        <div class="section">
            <div class="section-title">Information</div>
            <div class="row"><span class="label">Block ID</span><span class="value">${block.block_id || 'N/A'}</span></div>
            <div class="row"><span class="label">Offset</span><span class="value">${block.offset} (0x${block.offset?.toString(16) || '0'})</span></div>
            <div class="row"><span class="label">Size</span><span class="value">${formatSize(block.size)} (${block.size} bytes)</span></div>
            ${block.dataSize !== undefined ? `<div class="row"><span class="label">Data Size</span><span class="value">${formatSize(block.dataSize)} (${block.dataSize} bytes)</span></div>` : ''}
            <div class="row"><span class="label">Version</span><span class="value">${block.version || 1}</span></div>
            ${block.timestamp ? `<div class="row"><span class="label">Timestamp</span><span class="value">${formatTimestamp(block.timestamp)}</span></div>` : ''}
        </div>

        <div class="section">
            <div class="section-title">Content Hash</div>
            <div class="hash">${block.hash || 'N/A'}</div>
        </div>

        ${block.previous_hash && block.previous_hash !== '0'.repeat(64) ? `
        <div class="section">
            <div class="section-title">Previous Block Hash (Chain Link)</div>
            <div class="hash">${block.previous_hash}</div>
        </div>
        ` : ''}

        ${block.signature ? `
        <div class="section">
            <div class="section-title">Block Signature (RSA-PSS)</div>
            <div class="hash" style="font-size: 9px;">${block.signature}</div>
        </div>
        ` : ''}

        ${block.metadata ? `
        <div class="section">
            <div class="section-title">Metadata</div>
            <pre>${JSON.stringify(block.metadata, null, 2)}</pre>
        </div>
        ` : ''}
    </body>
    </html>`;
}
function formatAction(action) {
    const map = {
        'genesis': '🌟 Genesis',
        'add_text_block': ' Add Text Block',
        'add_embeddings_block': ' Add Embeddings',
        'add_knowledge_graph': ' Add Knowledge Graph',
        'add_image_block': ' Add Image',
        'add_audio_block': ' Add Audio',
        'add_video_block': ' Add Video',
        'finalize': ' Finalize',
        'sign': '✍️ Sign',
        'verify': ' Verify'
    };
    return map[action] || action;
}
function formatTimestamp(ts) {
    // Handle microseconds (secure format) vs seconds (legacy)
    const ms = ts > 1e12 ? ts / 1000 : ts * 1000;
    return new Date(ms).toLocaleString();
}
function truncate(str, len = 20) {
    if (!str)
        return 'N/A';
    return str.length > len * 2 ? `${str.slice(0, len)}...${str.slice(-len)}` : str;
}
function formatSize(bytes) {
    if (bytes === 0)
        return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
function deactivate() { }
//# sourceMappingURL=extension.js.map
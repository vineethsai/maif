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
exports.MAIFBinaryViewerProvider = void 0;
const vscode = __importStar(require("vscode"));
class MAIFBinaryViewerProvider {
    constructor(context) {
        this.context = context;
    }
    async openCustomDocument(uri, openContext, token) {
        return { uri, dispose: () => { } };
    }
    async resolveCustomEditor(document, webviewPanel, token) {
        webviewPanel.webview.options = {
            enableScripts: true
        };
        // Read the binary file
        const data = await vscode.workspace.fs.readFile(document.uri);
        const bytes = new Uint8Array(data);
        // Detect format and parse
        const fileInfo = this.detectFormat(bytes);
        const blocks = this.parseBlocks(bytes, fileInfo);
        // Generate hex dump
        const hexLines = this.generateHexDump(bytes);
        webviewPanel.webview.html = this.getHtmlContent(document.uri.fsPath, bytes.length, hexLines, blocks, fileInfo);
    }
    detectFormat(data) {
        if (data.length < 4) {
            return { format: 'legacy' };
        }
        const magic = String.fromCharCode(...data.slice(0, 4));
        if (magic === MAIFBinaryViewerProvider.MAGIC_HEADER && data.length >= MAIFBinaryViewerProvider.SECURE_HEADER_SIZE) {
            // Parse secure format header
            const view = new DataView(data.buffer);
            const versionMajor = view.getUint16(4, false);
            const versionMinor = view.getUint16(6, false);
            const flags = view.getUint32(8, false);
            const blockCount = view.getUint32(124, false);
            // Read agent DID (null-terminated string at offset 60)
            let agentDid = '';
            for (let i = 60; i < 124; i++) {
                if (data[i] === 0)
                    break;
                agentDid += String.fromCharCode(data[i]);
            }
            // Read merkle root (32 bytes at offset 152)
            const merkleRoot = this.bytesToHex(data.slice(152, 184));
            return {
                format: 'secure',
                version: `${versionMajor}.${versionMinor}`,
                agentDid,
                blockCount,
                merkleRoot,
                isSigned: !!(flags & 0x01),
                isFinalized: !!(flags & 0x02)
            };
        }
        return { format: 'legacy' };
    }
    parseBlocks(data, fileInfo) {
        if (fileInfo.format === 'secure') {
            return this.parseSecureBlocks(data, fileInfo.blockCount || 0);
        }
        return this.parseLegacyBlocks(data);
    }
    parseSecureBlocks(data, blockCount) {
        const blocks = [];
        let offset = MAIFBinaryViewerProvider.SECURE_HEADER_SIZE;
        for (let i = 0; i < blockCount && offset + MAIFBinaryViewerProvider.SECURE_BLOCK_HEADER_SIZE <= data.length; i++) {
            const view = new DataView(data.buffer, offset);
            const size = view.getUint32(0, false);
            const blockTypeCode = view.getUint32(4, false);
            const flags = view.getUint32(8, false);
            const metadataSize = view.getUint32(360, false);
            // Convert block type code to string
            const type = String.fromCharCode((blockTypeCode >> 24) & 0xFF, (blockTypeCode >> 16) & 0xFF, (blockTypeCode >> 8) & 0xFF, blockTypeCode & 0xFF);
            // Read block ID (16 bytes at offset 24)
            const blockId = this.bytesToHex(data.slice(offset + 24, offset + 40));
            // Read previous hash (32 bytes at offset 40)
            const previousHash = this.bytesToHex(data.slice(offset + 40, offset + 72));
            // Read content hash (32 bytes at offset 72)
            const hash = this.bytesToHex(data.slice(offset + 72, offset + 104));
            // Read signature (256 bytes at offset 104)
            const signature = this.bytesToHex(data.slice(offset + 104, offset + 360));
            const dataSize = size - MAIFBinaryViewerProvider.SECURE_BLOCK_HEADER_SIZE - metadataSize;
            blocks.push({
                type,
                offset,
                size,
                dataSize,
                blockId,
                isSigned: !!(flags & 0x01),
                hash,
                previousHash,
                signature
            });
            offset += size;
            if (blocks.length >= 100)
                break; // Safety limit
        }
        return blocks;
    }
    parseLegacyBlocks(data) {
        const blocks = [];
        let offset = 0;
        while (offset < data.length && offset + 32 <= data.length) {
            const view = new DataView(data.buffer, offset);
            const size = view.getUint32(0, false);
            if (size === 0 || size > data.length - offset)
                break;
            const typeBytes = data.slice(offset + 4, offset + 8);
            const type = String.fromCharCode(...typeBytes);
            blocks.push({ type, offset, size });
            offset += size;
            if (blocks.length >= 100)
                break; // Safety limit
        }
        return blocks;
    }
    generateHexDump(data, maxLines = 128) {
        const lines = [];
        const maxBytes = Math.min(data.length, maxLines * 16);
        for (let offset = 0; offset < maxBytes; offset += 16) {
            const lineBytes = data.slice(offset, Math.min(offset + 16, maxBytes));
            const offsetStr = offset.toString(16).padStart(8, '0');
            const hex = Array.from(lineBytes).map(b => b.toString(16).padStart(2, '0')).join(' ').padEnd(47, ' ');
            const ascii = Array.from(lineBytes).map(b => (b >= 32 && b < 127) ? String.fromCharCode(b) : '.').join('');
            lines.push(`<div class="hex-line"><span class="offset">${offsetStr}</span><span class="hex">${hex}</span><span class="ascii">${ascii}</span></div>`);
        }
        if (data.length > maxBytes) {
            lines.push(`<div class="more">... ${(data.length - maxBytes).toLocaleString()} more bytes ...</div>`);
        }
        return lines;
    }
    bytesToHex(bytes) {
        return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
    }
    formatSize(bytes) {
        if (bytes === 0)
            return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    truncateHash(hash, len = 16) {
        if (hash.length <= len * 2)
            return hash;
        return `${hash.slice(0, len)}...${hash.slice(-8)}`;
    }
    getHtmlContent(filename, totalBytes, hexLines, blocks, fileInfo) {
        const isSecure = fileInfo.format === 'secure';
        const blockHtml = blocks.map((b, i) => `
            <div class="block-item ${b.isSigned ? 'signed' : ''}" onclick="toggleBlockDetail(${i})">
                <div class="block-left">
                    <span class="block-type">${b.type}</span>
                    ${b.isSigned ? '<span class="signed-badge">🔐</span>' : ''}
                    <span class="expand-icon">▶</span>
                </div>
                <div class="block-right">
                    <div class="block-info">Offset: 0x${b.offset.toString(16)} | Size: ${this.formatSize(b.size)}${b.dataSize !== undefined ? ` | Data: ${this.formatSize(b.dataSize)}` : ''}</div>
                    ${b.blockId ? `<div class="block-id">ID: ${this.truncateHash(b.blockId)}</div>` : ''}
                    ${b.hash ? `<div class="block-hash">Hash: ${this.truncateHash(b.hash)}</div>` : ''}
                </div>
            </div>
            <div class="block-detail" id="block-detail-${i}">
                <div class="detail-section">
                    <div class="detail-title">Block Information</div>
                    <div class="detail-grid">
                        <div class="detail-row"><span class="detail-label">Type</span><span class="detail-value">${b.type}</span></div>
                        <div class="detail-row"><span class="detail-label">Index</span><span class="detail-value">#${i}</span></div>
                        <div class="detail-row"><span class="detail-label">Status</span><span class="detail-value">${b.isSigned ? '🔐 Signed & Immutable' : 'Unsigned'}</span></div>
                        <div class="detail-row"><span class="detail-label">Offset</span><span class="detail-value">${b.offset} (0x${b.offset.toString(16)})</span></div>
                        <div class="detail-row"><span class="detail-label">Total Size</span><span class="detail-value">${this.formatSize(b.size)} (${b.size} bytes)</span></div>
                        ${b.dataSize !== undefined ? `<div class="detail-row"><span class="detail-label">Data Size</span><span class="detail-value">${this.formatSize(b.dataSize)} (${b.dataSize} bytes)</span></div>` : ''}
                    </div>
                </div>
                ${b.blockId ? `
                <div class="detail-section">
                    <div class="detail-title">Block ID</div>
                    <div class="detail-hash">${b.blockId}</div>
                </div>
                ` : ''}
                ${b.hash ? `
                <div class="detail-section">
                    <div class="detail-title">Content Hash (SHA-256)</div>
                    <div class="detail-hash">${b.hash}</div>
                </div>
                ` : ''}
                ${b.previousHash && b.previousHash !== '0'.repeat(64) ? `
                <div class="detail-section">
                    <div class="detail-title">Previous Block Hash (Chain Link)</div>
                    <div class="detail-hash">${b.previousHash}</div>
                </div>
                ` : ''}
                ${b.signature ? `
                <div class="detail-section">
                    <div class="detail-title">Block Signature (RSA-2048-PSS)</div>
                    <div class="detail-hash signature">${b.signature}</div>
                </div>
                ` : ''}
            </div>
        `).join('');
        const formatBadge = isSecure
            ? '<span class="format-badge secure">🔒 Secure Format</span>'
            : '<span class="format-badge legacy">📁 Legacy Format</span>';
        const fileInfoHtml = isSecure ? `
            <div class="file-info">
                <div class="info-row"><span class="label">Version:</span> ${fileInfo.version}</div>
                <div class="info-row"><span class="label">Agent DID:</span> ${fileInfo.agentDid || 'N/A'}</div>
                <div class="info-row"><span class="label">Signed:</span> ${fileInfo.isSigned ? '✓ Yes' : '✗ No'}</div>
                <div class="info-row"><span class="label">Finalized:</span> ${fileInfo.isFinalized ? '✓ Yes' : '✗ No'}</div>
                ${fileInfo.merkleRoot ? `<div class="info-row"><span class="label">Merkle Root:</span> <code>${this.truncateHash(fileInfo.merkleRoot, 20)}</code></div>` : ''}
            </div>
        ` : '';
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MAIF Binary Viewer</title>
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    color: var(--vscode-foreground);
                    background: var(--vscode-editor-background);
                    padding: 20px;
                    margin: 0;
                }
                
                .header {
                    display: flex;
                    align-items: center;
                    gap: 15px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid var(--vscode-textSeparator-foreground);
                    margin-bottom: 20px;
                }
                
                .header-icon {
                    width: 48px;
                    height: 48px;
                    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    font-size: 16px;
                }
                
                .header-info h1 {
                    margin: 0 0 5px 0;
                    font-size: 18px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                
                .header-info .meta {
                    font-size: 13px;
                    color: var(--vscode-descriptionForeground);
                }
                
                .format-badge {
                    font-size: 11px;
                    padding: 3px 8px;
                    border-radius: 12px;
                    font-weight: 500;
                }
                
                .format-badge.secure {
                    background: rgba(16, 185, 129, 0.15);
                    color: #10b981;
                }
                
                .format-badge.legacy {
                    background: rgba(245, 158, 11, 0.15);
                    color: #f59e0b;
                }
                
                .file-info {
                    background: var(--vscode-editor-inactiveSelectionBackground);
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    font-size: 13px;
                }
                
                .file-info .info-row {
                    margin-bottom: 6px;
                }
                
                .file-info .label {
                    color: var(--vscode-descriptionForeground);
                    min-width: 100px;
                    display: inline-block;
                }
                
                .file-info code {
                    background: var(--vscode-textCodeBlock-background);
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 11px;
                }
                
                .tabs {
                    display: flex;
                    gap: 8px;
                    margin-bottom: 20px;
                }
                
                .tab {
                    padding: 8px 16px;
                    background: var(--vscode-button-secondaryBackground);
                    border: none;
                    border-radius: 6px;
                    color: var(--vscode-button-secondaryForeground);
                    cursor: pointer;
                    font-size: 13px;
                }
                
                .tab.active {
                    background: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                }
                
                .tab:hover:not(.active) {
                    background: var(--vscode-button-secondaryHoverBackground);
                }
                
                .panel {
                    display: none;
                }
                
                .panel.active {
                    display: block;
                }
                
                .hex-viewer {
                    font-family: var(--vscode-editor-font-family);
                    font-size: 13px;
                    line-height: 1.8;
                    background: var(--vscode-editor-inactiveSelectionBackground);
                    padding: 15px;
                    border-radius: 8px;
                    overflow: auto;
                    max-height: 600px;
                }
                
                .hex-line {
                    display: flex;
                    gap: 20px;
                }
                
                .offset {
                    color: var(--vscode-descriptionForeground);
                    min-width: 80px;
                }
                
                .hex {
                    color: var(--vscode-textLink-foreground);
                    min-width: 360px;
                }
                
                .ascii {
                    color: var(--vscode-foreground);
                }
                
                .more {
                    color: var(--vscode-descriptionForeground);
                    padding: 10px 0;
                    text-align: center;
                }
                
                .blocks-list {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                
                .block-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 12px 16px;
                    background: var(--vscode-editor-inactiveSelectionBackground);
                    border-radius: 8px;
                    transition: background 0.15s;
                }
                
                .block-item:hover {
                    background: var(--vscode-list-hoverBackground);
                }
                
                .block-item.signed {
                    border-left: 3px solid #10b981;
                }
                
                .block-item.expanded {
                    border-radius: 8px 8px 0 0;
                    margin-bottom: 0;
                }
                
                .block-item.expanded .expand-icon {
                    transform: rotate(90deg);
                }
                
                .expand-icon {
                    font-size: 10px;
                    color: var(--vscode-descriptionForeground);
                    transition: transform 0.2s ease;
                    margin-left: 8px;
                }
                
                .block-detail {
                    display: none;
                    background: var(--vscode-editor-background);
                    border: 1px solid var(--vscode-textSeparator-foreground);
                    border-top: none;
                    border-radius: 0 0 8px 8px;
                    padding: 16px;
                    margin-bottom: 10px;
                    animation: slideDown 0.2s ease;
                }
                
                .block-detail.show {
                    display: block;
                }
                
                @keyframes slideDown {
                    from { opacity: 0; transform: translateY(-10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                
                .detail-section {
                    margin-bottom: 16px;
                }
                
                .detail-section:last-child {
                    margin-bottom: 0;
                }
                
                .detail-title {
                    font-size: 11px;
                    text-transform: uppercase;
                    color: var(--vscode-descriptionForeground);
                    margin-bottom: 8px;
                    font-weight: 600;
                }
                
                .detail-grid {
                    display: grid;
                    gap: 6px;
                }
                
                .detail-row {
                    display: flex;
                    justify-content: space-between;
                    padding: 6px 10px;
                    background: var(--vscode-editor-inactiveSelectionBackground);
                    border-radius: 4px;
                    font-size: 13px;
                }
                
                .detail-label {
                    color: var(--vscode-descriptionForeground);
                }
                
                .detail-value {
                    font-family: var(--vscode-editor-font-family);
                    color: var(--vscode-foreground);
                }
                
                .detail-hash {
                    font-family: var(--vscode-editor-font-family);
                    font-size: 11px;
                    background: var(--vscode-textCodeBlock-background);
                    padding: 10px 12px;
                    border-radius: 6px;
                    word-break: break-all;
                    line-height: 1.6;
                    color: var(--vscode-textLink-foreground);
                }
                
                .detail-hash.signature {
                    font-size: 9px;
                    max-height: 100px;
                    overflow-y: auto;
                }
                
                .block-left {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .block-right {
                    text-align: right;
                }
                
                .block-type {
                    font-weight: 600;
                    padding: 4px 10px;
                    background: var(--vscode-badge-background);
                    color: var(--vscode-badge-foreground);
                    border-radius: 4px;
                    font-size: 12px;
                }
                
                .signed-badge {
                    font-size: 14px;
                }
                
                .block-info {
                    font-size: 13px;
                    color: var(--vscode-descriptionForeground);
                }
                
                .block-id, .block-hash {
                    font-size: 11px;
                    color: var(--vscode-descriptionForeground);
                    font-family: var(--vscode-editor-font-family);
                    margin-top: 4px;
                }
                
                .empty-state {
                    text-align: center;
                    padding: 40px;
                    color: var(--vscode-descriptionForeground);
                }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="header-icon">MAIF</div>
                <div class="header-info">
                    <h1>${filename.split('/').pop()} ${formatBadge}</h1>
                    <div class="meta">${this.formatSize(totalBytes)} • ${blocks.length} blocks detected</div>
                </div>
            </div>
            
            ${fileInfoHtml}
            
            <div class="tabs">
                <button class="tab active" onclick="showPanel('hex')">Hex View</button>
                <button class="tab" onclick="showPanel('blocks')">Blocks (${blocks.length})</button>
            </div>
            
            <div id="hex" class="panel active">
                <div class="hex-viewer">
                    ${hexLines.join('')}
                </div>
            </div>
            
            <div id="blocks" class="panel">
                ${blocks.length > 0 ? `
                    <div class="blocks-list">
                        ${blockHtml}
                    </div>
                ` : `
                    <div class="empty-state">
                        No blocks detected in binary structure
                    </div>
                `}
            </div>
            
            <script>
                function showPanel(panelId) {
                    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.getElementById(panelId).classList.add('active');
                    event.target.classList.add('active');
                }
                
                function toggleBlockDetail(index) {
                    const detail = document.getElementById('block-detail-' + index);
                    const blockItem = detail.previousElementSibling;
                    
                    // Close other open details
                    document.querySelectorAll('.block-detail.show').forEach(d => {
                        if (d.id !== 'block-detail-' + index) {
                            d.classList.remove('show');
                            d.previousElementSibling.classList.remove('expanded');
                        }
                    });
                    
                    // Toggle this one
                    detail.classList.toggle('show');
                    blockItem.classList.toggle('expanded');
                }
            </script>
        </body>
        </html>`;
    }
}
exports.MAIFBinaryViewerProvider = MAIFBinaryViewerProvider;
MAIFBinaryViewerProvider.viewType = 'maif.binaryViewer';
// Constants for secure format
MAIFBinaryViewerProvider.SECURE_HEADER_SIZE = 444;
MAIFBinaryViewerProvider.SECURE_BLOCK_HEADER_SIZE = 372;
MAIFBinaryViewerProvider.MAGIC_HEADER = 'MAIF';
//# sourceMappingURL=binaryViewer.js.map
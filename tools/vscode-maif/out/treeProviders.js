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
exports.MAIFProvenanceProvider = exports.MAIFBlocksProvider = exports.MAIFOverviewProvider = void 0;
const vscode = __importStar(require("vscode"));
const parser_1 = require("./parser");
// Tree item for overview
class OverviewItem extends vscode.TreeItem {
    constructor(label, value, collapsibleState = vscode.TreeItemCollapsibleState.None) {
        super(label, collapsibleState);
        this.label = label;
        this.value = value;
        this.collapsibleState = collapsibleState;
        this.description = value;
    }
}
// Overview Tree Provider
class MAIFOverviewProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this.manifest = null;
        this.fileInfo = null;
        this.securityInfo = null;
        this.format = 'unknown';
    }
    setManifest(manifest, fileInfo, securityInfo) {
        this.manifest = manifest;
        this.fileInfo = fileInfo || null;
        this.securityInfo = securityInfo || null;
        this.format = manifest.format || 'legacy';
        this._onDidChangeTreeData.fire();
    }
    refresh() {
        this._onDidChangeTreeData.fire();
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        if (!this.manifest) {
            return Promise.resolve([
                new OverviewItem('No MAIF file loaded', 'Open a .maif or manifest.json file')
            ]);
        }
        if (element) {
            return Promise.resolve([]);
        }
        const items = [];
        // Format indicator
        const isSecure = this.format === 'secure';
        items.push(new OverviewItem('Format', isSecure ? '🔒 Secure (Self-Contained)' : '📁 Legacy (+ Manifest)'));
        // Version
        items.push(new OverviewItem('Version', this.manifest.maif_version || this.manifest.header?.version || 'Unknown'));
        // Created
        items.push(new OverviewItem('Created', parser_1.MAIFParser.formatTimestamp(this.manifest.created || this.manifest.header?.created_timestamp || 0)));
        // Modified (secure format)
        if (isSecure && this.fileInfo?.modified) {
            items.push(new OverviewItem('Modified', parser_1.MAIFParser.formatTimestamp(this.fileInfo.modified)));
        }
        // Agent ID
        items.push(new OverviewItem('Agent ID', this.manifest.agent_id || this.manifest.header?.agent_id || 'N/A'));
        // File ID (secure format)
        if (isSecure && this.fileInfo?.fileId) {
            const fid = this.fileInfo.fileId;
            items.push(new OverviewItem('File ID', fid.length > 20 ? fid.slice(0, 10) + '...' + fid.slice(-8) : fid));
        }
        // Blocks
        items.push(new OverviewItem('Blocks', String(this.manifest.blocks?.length || 0)));
        // Signed
        items.push(new OverviewItem('Signed', this.manifest.signature ? '✓ Yes' : '✗ No'));
        // Finalized (secure format)
        if (isSecure && this.fileInfo) {
            items.push(new OverviewItem('Finalized', this.fileInfo.isFinalized ? '✓ Yes' : '✗ No'));
        }
        // Provenance
        items.push(new OverviewItem('Provenance', String(this.manifest.signature_metadata?.provenance_chain?.length || 0) + ' entries'));
        // Root Hash / Merkle Root
        const rootHash = this.manifest.merkle_root || this.manifest.root_hash;
        if (rootHash) {
            const truncated = rootHash.length > 30 ? rootHash.slice(0, 15) + '...' + rootHash.slice(-12) : rootHash;
            items.push(new OverviewItem(isSecure ? 'Merkle Root' : 'Root Hash', truncated));
        }
        return Promise.resolve(items);
    }
}
exports.MAIFOverviewProvider = MAIFOverviewProvider;
// Block tree item
class BlockItem extends vscode.TreeItem {
    constructor(block, index) {
        const statusIcons = [];
        if (block.isSigned)
            statusIcons.push('🔐');
        if (block.isTampered)
            statusIcons.push('⚠️');
        super(`${block.type} Block${statusIcons.length ? ' ' + statusIcons.join('') : ''}`, vscode.TreeItemCollapsibleState.None);
        this.block = block;
        this.index = index;
        const typeInfo = parser_1.MAIFParser.BLOCK_TYPES[block.type] || { name: block.type, icon: '?', color: '#64748b' };
        this.description = `${parser_1.MAIFParser.formatSize(block.size)} • v${block.version || 1}`;
        const tooltipLines = [
            `**${typeInfo.name} Block**`,
            ''
        ];
        if (block.isSigned || block.isTampered) {
            tooltipLines.push('### Status');
            if (block.isSigned)
                tooltipLines.push('- 🔐 **Signed**');
            if (block.isImmutable)
                tooltipLines.push('- 🔒 **Immutable**');
            if (block.isTampered)
                tooltipLines.push('- ⚠️ **TAMPERED**');
            tooltipLines.push('');
        }
        tooltipLines.push('### Details', `- **ID:** ${block.block_id || 'N/A'}`, `- **Size:** ${parser_1.MAIFParser.formatSize(block.size)}`, `- **Offset:** ${block.offset} (0x${block.offset.toString(16)})`, `- **Version:** ${block.version || 1}`, `- **Content Hash:** \`${(block.hash || 'N/A').slice(0, 20)}...\``);
        if (block.timestamp) {
            tooltipLines.push(`- **Timestamp:** ${parser_1.MAIFParser.formatTimestamp(block.timestamp)}`);
        }
        if (block.previous_hash && block.previous_hash !== '0'.repeat(64)) {
            tooltipLines.push(`- **Previous Hash:** \`${block.previous_hash.slice(0, 20)}...\``);
        }
        this.tooltip = new vscode.MarkdownString(tooltipLines.join('\n'));
        // Different icon based on status
        if (block.isTampered) {
            this.iconPath = new vscode.ThemeIcon('warning', new vscode.ThemeColor('errorForeground'));
        }
        else if (block.isSigned) {
            this.iconPath = new vscode.ThemeIcon('verified', new vscode.ThemeColor('charts.green'));
        }
        else {
            this.iconPath = new vscode.ThemeIcon('symbol-misc');
        }
        this.command = {
            command: 'maif.selectBlock',
            title: 'View Block Details',
            arguments: [index]
        };
    }
}
// Blocks Tree Provider
class MAIFBlocksProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this.blocks = [];
    }
    setBlocks(blocks) {
        this.blocks = blocks;
        this._onDidChangeTreeData.fire();
    }
    refresh() {
        this._onDidChangeTreeData.fire();
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        if (element) {
            return Promise.resolve([]);
        }
        if (this.blocks.length === 0) {
            return Promise.resolve([]);
        }
        return Promise.resolve(this.blocks.map((block, index) => new BlockItem(block, index)));
    }
}
exports.MAIFBlocksProvider = MAIFBlocksProvider;
// Provenance tree item
class ProvenanceItem extends vscode.TreeItem {
    constructor(entry, index) {
        const isGenesis = entry.action === 'genesis';
        super(formatAction(entry.action), vscode.TreeItemCollapsibleState.None);
        this.entry = entry;
        this.index = index;
        this.description = parser_1.MAIFParser.formatTimestamp(entry.timestamp);
        this.tooltip = new vscode.MarkdownString([
            `**${formatAction(entry.action)}**`,
            '',
            `- **Agent:** ${entry.agent_id}`,
            `- **DID:** ${entry.agent_did || 'N/A'}`,
            `- **Time:** ${parser_1.MAIFParser.formatTimestamp(entry.timestamp)}`,
            `- **Entry Hash:** \`${(entry.entry_hash || 'N/A').slice(0, 20)}...\``,
            `- **Block Hash:** \`${(entry.block_hash || 'N/A').slice(0, 20)}...\``,
            entry.signature ? `- **Signed:** ✓` : ''
        ].join('\n'));
        this.iconPath = new vscode.ThemeIcon(isGenesis ? 'star' : 'git-commit');
    }
}
// Provenance Tree Provider
class MAIFProvenanceProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this.provenance = [];
    }
    setProvenance(provenance) {
        this.provenance = provenance;
        this._onDidChangeTreeData.fire();
    }
    refresh() {
        this._onDidChangeTreeData.fire();
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        if (element) {
            return Promise.resolve([]);
        }
        if (this.provenance.length === 0) {
            return Promise.resolve([]);
        }
        return Promise.resolve(this.provenance.map((entry, index) => new ProvenanceItem(entry, index)));
    }
}
exports.MAIFProvenanceProvider = MAIFProvenanceProvider;
function formatAction(action) {
    const map = {
        'genesis': '🌟 Genesis',
        'add_text_block': '📝 Add Text',
        'add_embeddings_block': '🧠 Add Embeddings',
        'add_knowledge_graph': '🕸️ Knowledge Graph',
        'add_image_block': '🖼️ Add Image',
        'add_audio_block': '🎵 Add Audio',
        'add_video_block': '🎬 Add Video',
        'update': '✏️ Update',
        'delete': '🗑️ Delete',
        'finalize': '🔒 Finalize',
        'sign': '✍️ Sign',
        'verify': '✅ Verify'
    };
    return map[action] || action;
}
//# sourceMappingURL=treeProviders.js.map
import * as vscode from 'vscode';
import { MAIFBlock, MAIFManifest, MAIFProvenanceEntry, MAIFParser, MAIFFileInfo, MAIFSecurityInfo } from './parser';

// Tree item for overview
class OverviewItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly value: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState = vscode.TreeItemCollapsibleState.None
    ) {
        super(label, collapsibleState);
        this.description = value;
    }
}

// Overview Tree Provider
export class MAIFOverviewProvider implements vscode.TreeDataProvider<OverviewItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<OverviewItem | undefined | null | void> = new vscode.EventEmitter<OverviewItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<OverviewItem | undefined | null | void> = this._onDidChangeTreeData.event;

    private manifest: MAIFManifest | null = null;
    private fileInfo: MAIFFileInfo | null = null;
    private securityInfo: MAIFSecurityInfo | null = null;
    private format: 'legacy' | 'secure' | 'unknown' = 'unknown';

    setManifest(manifest: MAIFManifest, fileInfo?: MAIFFileInfo, securityInfo?: MAIFSecurityInfo): void {
        this.manifest = manifest;
        this.fileInfo = fileInfo || null;
        this.securityInfo = securityInfo || null;
        this.format = manifest.format || 'legacy';
        this._onDidChangeTreeData.fire();
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: OverviewItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: OverviewItem): Thenable<OverviewItem[]> {
        if (!this.manifest) {
            return Promise.resolve([
                new OverviewItem('No MAIF file loaded', 'Open a .maif or manifest.json file')
            ]);
        }

        if (element) {
            return Promise.resolve([]);
        }

        const items: OverviewItem[] = [];
        
        // Format indicator
        const isSecure = this.format === 'secure';
        items.push(new OverviewItem('Format', isSecure ? '🔒 Secure (Self-Contained)' : '📁 Legacy (+ Manifest)'));

        // Version
        items.push(new OverviewItem('Version', this.manifest.maif_version || this.manifest.header?.version || 'Unknown'));
        
        // Created
        items.push(new OverviewItem('Created', MAIFParser.formatTimestamp(this.manifest.created || this.manifest.header?.created_timestamp || 0)));
        
        // Modified (secure format)
        if (isSecure && this.fileInfo?.modified) {
            items.push(new OverviewItem('Modified', MAIFParser.formatTimestamp(this.fileInfo.modified)));
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

// Block tree item
class BlockItem extends vscode.TreeItem {
    constructor(
        public readonly block: MAIFBlock,
        public readonly index: number
    ) {
        const statusIcons: string[] = [];
        if (block.isSigned) statusIcons.push('🔐');
        if (block.isTampered) statusIcons.push('⚠️');
        
        super(`${block.type} Block${statusIcons.length ? ' ' + statusIcons.join('') : ''}`, vscode.TreeItemCollapsibleState.None);
        
        const typeInfo = MAIFParser.BLOCK_TYPES[block.type] || { name: block.type, icon: '?', color: '#64748b' };
        
        this.description = `${MAIFParser.formatSize(block.size)} • v${block.version || 1}`;
        
        const tooltipLines = [
            `**${typeInfo.name} Block**`,
            ''
        ];
        
        if (block.isSigned || block.isTampered) {
            tooltipLines.push('### Status');
            if (block.isSigned) tooltipLines.push('- 🔐 **Signed**');
            if (block.isImmutable) tooltipLines.push('- 🔒 **Immutable**');
            if (block.isTampered) tooltipLines.push('- ⚠️ **TAMPERED**');
            tooltipLines.push('');
        }
        
        tooltipLines.push(
            '### Details',
            `- **ID:** ${block.block_id || 'N/A'}`,
            `- **Size:** ${MAIFParser.formatSize(block.size)}`,
            `- **Offset:** ${block.offset} (0x${block.offset.toString(16)})`,
            `- **Version:** ${block.version || 1}`,
            `- **Content Hash:** \`${(block.hash || 'N/A').slice(0, 20)}...\``
        );
        
        if (block.timestamp) {
            tooltipLines.push(`- **Timestamp:** ${MAIFParser.formatTimestamp(block.timestamp)}`);
        }
        
        if (block.previous_hash && block.previous_hash !== '0'.repeat(64)) {
            tooltipLines.push(`- **Previous Hash:** \`${block.previous_hash.slice(0, 20)}...\``);
        }
        
        this.tooltip = new vscode.MarkdownString(tooltipLines.join('\n'));
        
        // Different icon based on status
        if (block.isTampered) {
            this.iconPath = new vscode.ThemeIcon('warning', new vscode.ThemeColor('errorForeground'));
        } else if (block.isSigned) {
            this.iconPath = new vscode.ThemeIcon('verified', new vscode.ThemeColor('charts.green'));
        } else {
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
export class MAIFBlocksProvider implements vscode.TreeDataProvider<BlockItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<BlockItem | undefined | null | void> = new vscode.EventEmitter<BlockItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<BlockItem | undefined | null | void> = this._onDidChangeTreeData.event;

    private blocks: MAIFBlock[] = [];

    setBlocks(blocks: MAIFBlock[]): void {
        this.blocks = blocks;
        this._onDidChangeTreeData.fire();
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: BlockItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: BlockItem): Thenable<BlockItem[]> {
        if (element) {
            return Promise.resolve([]);
        }

        if (this.blocks.length === 0) {
            return Promise.resolve([]);
        }

        return Promise.resolve(
            this.blocks.map((block, index) => new BlockItem(block, index))
        );
    }
}

// Provenance tree item
class ProvenanceItem extends vscode.TreeItem {
    constructor(
        public readonly entry: MAIFProvenanceEntry,
        public readonly index: number
    ) {
        const isGenesis = entry.action === 'genesis';
        super(
            formatAction(entry.action),
            vscode.TreeItemCollapsibleState.None
        );
        
        this.description = MAIFParser.formatTimestamp(entry.timestamp);
        this.tooltip = new vscode.MarkdownString([
            `**${formatAction(entry.action)}**`,
            '',
            `- **Agent:** ${entry.agent_id}`,
            `- **DID:** ${entry.agent_did || 'N/A'}`,
            `- **Time:** ${MAIFParser.formatTimestamp(entry.timestamp)}`,
            `- **Entry Hash:** \`${(entry.entry_hash || 'N/A').slice(0, 20)}...\``,
            `- **Block Hash:** \`${(entry.block_hash || 'N/A').slice(0, 20)}...\``,
            entry.signature ? `- **Signed:** ✓` : ''
        ].join('\n'));
        
        this.iconPath = new vscode.ThemeIcon(isGenesis ? 'star' : 'git-commit');
    }
}

// Provenance Tree Provider
export class MAIFProvenanceProvider implements vscode.TreeDataProvider<ProvenanceItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<ProvenanceItem | undefined | null | void> = new vscode.EventEmitter<ProvenanceItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<ProvenanceItem | undefined | null | void> = this._onDidChangeTreeData.event;

    private provenance: MAIFProvenanceEntry[] = [];

    setProvenance(provenance: MAIFProvenanceEntry[]): void {
        this.provenance = provenance;
        this._onDidChangeTreeData.fire();
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: ProvenanceItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ProvenanceItem): Thenable<ProvenanceItem[]> {
        if (element) {
            return Promise.resolve([]);
        }

        if (this.provenance.length === 0) {
            return Promise.resolve([]);
        }

        return Promise.resolve(
            this.provenance.map((entry, index) => new ProvenanceItem(entry, index))
        );
    }
}

function formatAction(action: string): string {
    const map: Record<string, string> = {
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


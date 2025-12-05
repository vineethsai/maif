/**
 * MAIF Explorer Application
 * Interactive viewer for MAIF files
 */

class MAIFExplorer {
    constructor() {
        this.parser = new MAIFParser();
        this.theme = localStorage.getItem('theme') || 'dark';
        this.selectedBlock = null;
        
        this.init();
    }

    init() {
        this.setupTheme();
        this.setupEventListeners();
        this.setupDragDrop();
    }

    setupTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
    }

    setupEventListeners() {
        // Theme toggle
        document.getElementById('themeToggle')?.addEventListener('click', () => {
            this.theme = this.theme === 'dark' ? 'light' : 'dark';
            localStorage.setItem('theme', this.theme);
            this.setupTheme();
        });

        // File input
        document.getElementById('browseBtn')?.addEventListener('click', () => {
            document.getElementById('fileInput')?.click();
        });

        document.getElementById('fileInput')?.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });

        // Load sample
        document.getElementById('loadSampleBtn')?.addEventListener('click', () => {
            this.loadSample();
        });

        // Close file
        document.getElementById('closeFileBtn')?.addEventListener('click', () => {
            this.closeFile();
        });

        // Open file button (in header)
        document.getElementById('openFileBtn')?.addEventListener('click', () => {
            document.getElementById('fileInput')?.click();
        });

        // Tab navigation
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Close detail panel
        document.getElementById('closeDetailBtn')?.addEventListener('click', () => {
            this.closeDetailPanel();
        });

        // Block search
        document.getElementById('blockSearch')?.addEventListener('input', (e) => {
            this.filterBlocks(e.target.value);
        });

        // Block type filter
        document.getElementById('blockTypeFilter')?.addEventListener('change', (e) => {
            this.filterBlocksByType(e.target.value);
        });

        // Copy hex button
        document.getElementById('copyHexBtn')?.addEventListener('click', () => {
            this.copyHexToClipboard();
        });
    }

    setupDragDrop() {
        const dropZone = document.getElementById('dropZone');
        if (!dropZone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove('drag-over');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            this.handleFiles(files);
        });
    }

    async handleFiles(files) {
        if (!files || files.length === 0) return;

        let manifestFile = null;
        let binaryFile = null;

        for (const file of files) {
            if (file.name.endsWith('.json')) {
                manifestFile = file;
            } else if (file.name.endsWith('.maif')) {
                binaryFile = file;
            }
        }

        if (manifestFile) {
            try {
                await this.parser.loadManifest(manifestFile);
            } catch (err) {
                this.showError('Failed to load manifest: ' + err.message);
                return;
            }
        }

        if (binaryFile) {
            try {
                await this.parser.loadBinary(binaryFile);
            } catch (err) {
                this.showError('Failed to load binary: ' + err.message);
                return;
            }
        }

        if (this.parser.manifest || this.parser.binaryData) {
            this.showExplorer();
            this.renderContent();
        } else {
            this.showError('Please provide a .maif file and/or its manifest.json');
        }
    }

    loadSample() {
        this.parser.manifest = window.SAMPLE_MANIFEST;
        this.parser.binaryData = window.generateSampleBinary();
        this.parser.parseManifest();
        this.parser.parseBinary();
        
        this.showExplorer();
        this.renderContent();
    }

    showExplorer() {
        document.getElementById('dropZone')?.classList.add('hidden');
        document.getElementById('explorerContent')?.classList.remove('hidden');
        document.getElementById('openFileBtn')?.style.setProperty('display', 'flex');
    }

    closeFile() {
        this.parser = new MAIFParser();
        this.selectedBlock = null;
        
        document.getElementById('dropZone')?.classList.remove('hidden');
        document.getElementById('explorerContent')?.classList.add('hidden');
        document.getElementById('openFileBtn')?.style.setProperty('display', 'none');
        this.closeDetailPanel();
    }

    switchTab(tabId) {
        // Update tab buttons
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabId);
        });

        // Update tab panes
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.toggle('active', pane.id === tabId);
        });
    }

    renderContent() {
        this.renderFileTree();
        this.renderOverview();
        this.renderBlocks();
        this.renderProvenance();
        this.renderSecurity();
        this.renderBinary();
    }

    renderFileTree() {
        const tree = document.getElementById('fileTree');
        if (!tree) return;

        const overview = this.parser.getOverview();
        if (!overview) return;

        tree.innerHTML = `
            <div class="tree-item selected" data-view="overview">
                <svg class="tree-icon icon" viewBox="0 0 24 24">
                    <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                </svg>
                <span class="tree-label">Overview</span>
            </div>
            <div class="tree-item" data-view="blocks">
                <svg class="tree-icon icon" viewBox="0 0 24 24">
                    <path d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
                </svg>
                <span class="tree-label">Blocks</span>
                <span class="tree-badge">${overview.totalBlocks}</span>
            </div>
            <div class="tree-item" data-view="provenance">
                <svg class="tree-icon icon" viewBox="0 0 24 24">
                    <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/>
                </svg>
                <span class="tree-label">Provenance</span>
                <span class="tree-badge">${this.parser.provenance.length}</span>
            </div>
            <div class="tree-item" data-view="security">
                <svg class="tree-icon icon" viewBox="0 0 24 24">
                    <path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
                </svg>
                <span class="tree-label">Security</span>
            </div>
        `;

        // Add click handlers
        tree.querySelectorAll('.tree-item').forEach(item => {
            item.addEventListener('click', () => {
                tree.querySelectorAll('.tree-item').forEach(i => i.classList.remove('selected'));
                item.classList.add('selected');
                this.switchTab(item.dataset.view);
            });
        });
    }

    renderOverview() {
        const overview = this.parser.getOverview();
        const signature = this.parser.getSignatureInfo();
        
        if (!overview) return;

        const isSecure = overview.format === 'secure';

        // File info
        const fileInfo = document.getElementById('fileInfoContent');
        if (fileInfo) {
            fileInfo.innerHTML = `
                <div class="info-row">
                    <span class="info-label">Format</span>
                    <span class="info-value">
                        <span class="status-badge ${isSecure ? 'valid' : 'warning'}" style="font-size: 11px; padding: 2px 8px;">
                            ${isSecure ? ' Secure' : '📁 Legacy'}
                        </span>
                    </span>
                </div>
                <div class="info-row">
                    <span class="info-label">Version</span>
                    <span class="info-value">${overview.version}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Created</span>
                    <span class="info-value">${MAIFParser.formatTimestamp(overview.created)}</span>
                </div>
                ${overview.modified && isSecure ? `
                    <div class="info-row">
                        <span class="info-label">Modified</span>
                        <span class="info-value">${MAIFParser.formatTimestamp(overview.modified)}</span>
                    </div>
                ` : ''}
                <div class="info-row">
                    <span class="info-label">Agent ID</span>
                    <span class="info-value mono" style="font-size: 11px;">${overview.agentId || 'N/A'}</span>
                </div>
                ${isSecure && overview.fileId ? `
                    <div class="info-row">
                        <span class="info-label">File ID</span>
                        <span class="info-value mono" style="font-size: 11px;">${this.truncateHash(overview.fileId)}</span>
                    </div>
                ` : ''}
                <div class="info-row">
                    <span class="info-label">Total Size</span>
                    <span class="info-value">${MAIFParser.formatSize(overview.totalSize)}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">${isSecure ? 'Merkle Root' : 'Root Hash'}</span>
                    <span class="info-value mono" style="font-size: 10px;">${this.truncateHash(overview.rootHash)}</span>
                </div>
            `;
        }

        // Signature status
        const sigStatus = document.getElementById('signatureStatusContent');
        if (sigStatus) {
            const hasSignature = overview.hasSignature;
            const isFinalized = overview.isFinalized;
            const tamperedCount = overview.tamperedBlocks || 0;
            
            sigStatus.innerHTML = `
                <div style="text-align: center; padding: 20px 0;">
                    <div class="status-badge ${hasSignature ? 'valid' : 'warning'}" style="font-size: 14px; padding: 8px 16px;">
                        <span class="status-dot"></span>
                        ${hasSignature ? 'Signed' : 'Unsigned'}
                    </div>
                    ${isSecure ? `
                        <div style="margin-top: 8px;">
                            <span class="status-badge ${isFinalized ? 'valid' : 'warning'}" style="font-size: 11px; padding: 4px 8px;">
                                ${isFinalized ? ' Finalized' : '⏳ Not Finalized'}
                            </span>
                        </div>
                    ` : ''}
                    ${tamperedCount > 0 ? `
                        <div style="margin-top: 8px;">
                            <span class="status-badge invalid" style="font-size: 11px; padding: 4px 8px;">
                                 ${tamperedCount} Tampered Block${tamperedCount > 1 ? 's' : ''}
                            </span>
                        </div>
                    ` : ''}
                    ${hasSignature ? `
                        <div style="margin-top: 16px; font-size: 13px; color: var(--text-secondary);">
                            <div>Signer: <strong>${signature?.signerId || signature?.signerDid || 'Unknown'}</strong></div>
                            <div style="margin-top: 4px;">Signed: ${MAIFParser.formatTimestamp(signature?.timestamp)}</div>
                            ${signature?.algorithm ? `<div style="margin-top: 4px; font-size: 11px; color: var(--text-muted);">Algorithm: ${signature.algorithm}</div>` : ''}
                        </div>
                    ` : ''}
                </div>
            `;
        }

        // Block stats
        const blockStats = document.getElementById('blockStatsContent');
        if (blockStats) {
            const stats = overview.blockStats;
            blockStats.innerHTML = Object.entries(stats).map(([type, data]) => {
                const typeInfo = MAIFParser.BLOCK_TYPES[type] || { name: type, icon: '?', color: '#64748b' };
                return `
                    <div class="block-stat">
                        <div class="block-stat-type">
                            <div class="block-type-icon ${type.toLowerCase()}">${typeInfo.icon}</div>
                            <span>${typeInfo.name}</span>
                        </div>
                        <div>
                            <span class="block-stat-count">${data.count}</span>
                            <span style="font-size: 12px; color: var(--text-muted); margin-left: 4px;">
                                (${MAIFParser.formatSize(data.totalSize)})
                            </span>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Integrity
        const integrity = document.getElementById('integrityContent');
        if (integrity) {
            integrity.innerHTML = `
                <div style="text-align: center; padding: 20px 0;">
                    <div class="status-badge valid" style="font-size: 14px; padding: 8px 16px;">
                        <span class="status-dot"></span>
                        File Loaded Successfully
                    </div>
                    <div style="margin-top: 16px; font-size: 13px; color: var(--text-secondary);">
                        ${overview.totalBlocks} blocks parsed<br>
                        ${overview.hasProvenance ? `${this.parser.provenance.length} provenance entries` : 'No provenance data'}
                    </div>
                </div>
            `;
        }
    }

    renderBlocks() {
        const grid = document.getElementById('blocksGrid');
        if (!grid) return;

        const isSecureFormat = this.parser.format === 'secure';

        grid.innerHTML = this.parser.blocks.map((block, index) => {
            const typeInfo = block.typeInfo;
            return `
                <div class="block-card ${block.isTampered ? 'tampered' : ''}" data-index="${index}">
                    <div class="block-card-header">
                        <div class="block-type-icon ${block.type.toLowerCase()}">${typeInfo.icon}</div>
                        <div style="flex: 1;">
                            <div style="font-weight: 600; display: flex; align-items: center; gap: 8px;">
                                ${typeInfo.name} Block
                                ${isSecureFormat && block.isSigned ? '<span style="font-size: 10px;"></span>' : ''}
                                ${block.isTampered ? '<span style="font-size: 10px;"></span>' : ''}
                            </div>
                            <div style="font-size: 12px; color: var(--text-muted);">
                                ${block.block_id ? this.truncateHash(block.block_id) : `Block #${index}`}
                            </div>
                        </div>
                    </div>
                    <div class="block-card-body">
                        <div class="block-card-meta">
                            <div class="block-meta-item">
                                <svg class="icon" style="width: 14px; height: 14px;" viewBox="0 0 24 24">
                                    <path d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7"/>
                                </svg>
                                ${MAIFParser.formatSize(block.size || 0)}
                            </div>
                            <div class="block-meta-item">
                                <svg class="icon" style="width: 14px; height: 14px;" viewBox="0 0 24 24">
                                    <path d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"/>
                                </svg>
                                v${block.version || 1}
                            </div>
                            ${block.metadata?.source ? `
                                <div class="block-meta-item">
                                    <svg class="icon" style="width: 14px; height: 14px;" viewBox="0 0 24 24">
                                        <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                                    </svg>
                                    ${block.metadata.source}
                                </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Add click handlers
        grid.querySelectorAll('.block-card').forEach(card => {
            card.addEventListener('click', () => {
                const index = parseInt(card.dataset.index);
                this.showBlockDetail(index);
            });
        });
    }

    showBlockDetail(index) {
        const block = this.parser.blocks[index];
        if (!block) return;

        this.selectedBlock = index;
        
        const panel = document.getElementById('detailPanel');
        const content = document.getElementById('detailContent');
        const title = document.getElementById('detailTitle');
        const explorer = document.getElementById('explorerContent');

        if (!panel || !content || !title) return;

        title.textContent = `${block.typeInfo.name} Block`;
        
        const isSecureFormat = this.parser.format === 'secure';
        
        content.innerHTML = `
            <div class="detail-section">
                <div class="detail-section-title">Block Information</div>
                ${isSecureFormat ? `
                    <div class="info-row" style="margin-bottom: 12px;">
                        <span class="info-label">Status</span>
                        <span class="info-value">
                            ${block.isSigned ? '<span class="status-badge valid" style="font-size: 10px; padding: 2px 6px;"> Signed</span>' : ''}
                            ${block.isImmutable ? '<span class="status-badge valid" style="font-size: 10px; padding: 2px 6px; margin-left: 4px;"> Immutable</span>' : ''}
                            ${block.isTampered ? '<span class="status-badge invalid" style="font-size: 10px; padding: 2px 6px; margin-left: 4px;"> TAMPERED</span>' : ''}
                        </span>
                    </div>
                ` : ''}
                <div class="info-row">
                    <span class="info-label">Type</span>
                    <span class="info-value">${block.type}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Block ID</span>
                    <span class="info-value mono" style="font-size: 11px;">${block.block_id || 'N/A'}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Offset</span>
                    <span class="info-value">${block.offset} (0x${block.offset?.toString(16) || '0'})</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Size</span>
                    <span class="info-value">${MAIFParser.formatSize(block.size || 0)} (${block.size} bytes)</span>
                </div>
                ${isSecureFormat && block.dataSize !== undefined ? `
                    <div class="info-row">
                        <span class="info-label">Data Size</span>
                        <span class="info-value">${MAIFParser.formatSize(block.dataSize)} (${block.dataSize} bytes)</span>
                    </div>
                ` : ''}
                <div class="info-row">
                    <span class="info-label">Version</span>
                    <span class="info-value">${block.version || 1}</span>
                </div>
                ${block.timestamp ? `
                    <div class="info-row">
                        <span class="info-label">Timestamp</span>
                        <span class="info-value">${MAIFParser.formatTimestamp(block.timestamp)}</span>
                    </div>
                ` : ''}
            </div>

            <div class="detail-section">
                <div class="detail-section-title">Content Hash</div>
                <div class="signature-hash">${block.hash || block.content_hash || 'N/A'}</div>
            </div>

            ${block.previous_hash && block.previous_hash !== '0'.repeat(64) ? `
                <div class="detail-section">
                    <div class="detail-section-title">Previous Block Hash (Chain Link)</div>
                    <div class="signature-hash">${block.previous_hash}</div>
                </div>
            ` : ''}

            ${isSecureFormat && block.signature ? `
                <div class="detail-section">
                    <div class="detail-section-title">Block Signature (RSA-PSS)</div>
                    <div class="signature-hash" style="word-break: break-all; font-size: 9px;">${block.signature}</div>
                </div>
            ` : ''}

            ${block.metadata && Object.keys(block.metadata).length > 0 ? `
                <div class="detail-section">
                    <div class="detail-section-title">Metadata</div>
                    <pre class="key-display">${JSON.stringify(block.metadata, null, 2)}</pre>
                </div>
            ` : ''}

            ${this.getContentPreviewHtml(block, index)}
        `;

        panel.classList.remove('hidden');
        explorer?.classList.add('has-detail');
    }

    closeDetailPanel() {
        const panel = document.getElementById('detailPanel');
        const explorer = document.getElementById('explorerContent');
        
        panel?.classList.add('hidden');
        explorer?.classList.remove('has-detail');
        this.selectedBlock = null;
    }

    renderProvenance() {
        const timeline = document.getElementById('provenanceTimeline');
        const count = document.getElementById('provenanceCount');
        
        if (!timeline) return;

        const entries = this.parser.provenance;
        
        if (count) {
            count.textContent = `${entries.length} entries`;
        }

        if (entries.length === 0) {
            timeline.innerHTML = `
                <div style="text-align: center; padding: 40px; color: var(--text-muted);">
                    No provenance data available
                </div>
            `;
            return;
        }

        timeline.innerHTML = entries.map((entry, index) => {
            const isGenesis = entry.action === 'genesis';
            return `
                <div class="timeline-entry ${isGenesis ? 'genesis' : ''} slide-in" style="animation-delay: ${index * 50}ms;">
                    <div class="timeline-header">
                        <span class="timeline-action">${this.formatAction(entry.action)}</span>
                        <span class="timeline-time">${MAIFParser.formatTimestamp(entry.timestamp)}</span>
                    </div>
                    <div class="timeline-details">
                        <div class="timeline-detail">
                            <span class="timeline-detail-label">Agent</span>
                            <span class="timeline-detail-value">${entry.agent_id || 'Unknown'}</span>
                        </div>
                        <div class="timeline-detail">
                            <span class="timeline-detail-label">DID</span>
                            <span class="timeline-detail-value">${entry.agent_did || 'N/A'}</span>
                        </div>
                        <div class="timeline-detail">
                            <span class="timeline-detail-label">Entry Hash</span>
                            <span class="timeline-detail-value">${this.truncateHash(entry.entry_hash)}</span>
                        </div>
                        <div class="timeline-detail">
                            <span class="timeline-detail-label">Block Hash</span>
                            <span class="timeline-detail-value">${this.truncateHash(entry.block_hash)}</span>
                        </div>
                    </div>
                    ${entry.signature ? `
                        <div class="timeline-signature">
                            <div class="timeline-detail-label">Signature</div>
                            <div class="signature-preview">${this.truncateSignature(entry.signature)}</div>
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');
    }

    renderSecurity() {
        const signature = this.parser.getSignatureInfo();
        const overview = this.parser.getOverview();
        const isSecureFormat = this.parser.format === 'secure';
        
        // Public key
        const keyDisplay = document.getElementById('publicKeyDisplay');
        if (keyDisplay) {
            keyDisplay.textContent = signature?.publicKey || 'No public key available';
        }

        // Signature info
        const sigInfo = document.getElementById('signatureInfo');
        if (sigInfo && signature) {
            sigInfo.innerHTML = `
                <div style="margin-bottom: 12px;">
                    <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 4px;">Status</div>
                    <span class="status-badge ${signature.signature ? 'valid' : 'warning'}">
                        <span class="status-dot"></span>
                        ${signature.signature ? 'Present' : 'Missing'}
                    </span>
                    ${isSecureFormat ? `
                        <span class="status-badge valid" style="margin-left: 8px;">
                             Self-Contained
                        </span>
                    ` : `
                        <span class="status-badge warning" style="margin-left: 8px;">
                            📁 External Manifest
                        </span>
                    `}
                </div>
                ${signature.signature ? `
                    <div style="margin-top: 12px;">
                        <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 4px;">File Signature</div>
                        <div class="signature-hash" style="font-size: 9px;">${signature.signature}</div>
                    </div>
                ` : '<div style="color: var(--text-muted);">No signature found</div>'}
            `;
        }

        // Signer info
        const signerInfo = document.getElementById('signerInfo');
        if (signerInfo && signature) {
            signerInfo.innerHTML = `
                <div class="info-row">
                    <span class="info-label">Signer ID</span>
                    <span class="info-value">${signature.signerId || 'Unknown'}</span>
                </div>
                ${signature.signerDid ? `
                    <div class="info-row">
                        <span class="info-label">Signer DID</span>
                        <span class="info-value mono" style="font-size: 10px;">${signature.signerDid}</span>
                    </div>
                ` : ''}
                <div class="info-row">
                    <span class="info-label">Signed At</span>
                    <span class="info-value">${MAIFParser.formatTimestamp(signature.timestamp)}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Algorithm</span>
                    <span class="info-value">${signature.algorithm || 'RSA-PSS with SHA-256'}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Key Algorithm</span>
                    <span class="info-value">${signature.keyAlgorithm || 'RSA-2048'}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Provenance Entries</span>
                    <span class="info-value">${this.parser.provenance.length}</span>
                </div>
                ${isSecureFormat ? `
                    <div class="info-row">
                        <span class="info-label">Signed Blocks</span>
                        <span class="info-value">${this.parser.blocks.filter(b => b.isSigned).length} / ${this.parser.blocks.length}</span>
                    </div>
                ` : ''}
            `;
        }
    }

    renderBinary() {
        const viewer = document.getElementById('hexViewer');
        const info = document.getElementById('binaryInfo');
        
        if (!viewer) return;

        if (!this.parser.binaryData) {
            viewer.innerHTML = '<div style="color: var(--text-muted);">No binary data loaded</div>';
            return;
        }

        if (info) {
            info.textContent = `${MAIFParser.formatSize(this.parser.binaryData.length)} | ${this.parser.binaryData.length.toLocaleString()} bytes`;
        }

        const lines = this.parser.getHexDump(0, 1024);
        viewer.innerHTML = lines.map(line => `
            <div class="hex-line">
                <span class="hex-offset">${line.offset}</span>
                <span class="hex-bytes">${line.hex}</span>
                <span class="hex-ascii">${line.ascii}</span>
            </div>
        `).join('');
    }

    filterBlocks(query) {
        const cards = document.querySelectorAll('.block-card');
        const lowerQuery = query.toLowerCase();

        cards.forEach(card => {
            const index = parseInt(card.dataset.index);
            const block = this.parser.blocks[index];
            
            const matches = 
                block.type.toLowerCase().includes(lowerQuery) ||
                block.block_id?.toLowerCase().includes(lowerQuery) ||
                JSON.stringify(block.metadata || {}).toLowerCase().includes(lowerQuery);

            card.style.display = matches ? '' : 'none';
        });
    }

    filterBlocksByType(type) {
        const cards = document.querySelectorAll('.block-card');

        cards.forEach(card => {
            const index = parseInt(card.dataset.index);
            const block = this.parser.blocks[index];
            
            const matches = !type || block.type === type;
            card.style.display = matches ? '' : 'none';
        });
    }

    copyHexToClipboard() {
        const lines = this.parser.getHexDump(0, this.parser.binaryData?.length || 0);
        const text = lines.map(l => `${l.offset}  ${l.hex}  ${l.ascii}`).join('\n');
        
        navigator.clipboard.writeText(text).then(() => {
            const btn = document.getElementById('copyHexBtn');
            if (btn) {
                const original = btn.textContent;
                btn.textContent = 'Copied!';
                setTimeout(() => { btn.textContent = original; }, 2000);
            }
        });
    }

    getContentPreviewHtml(block, index) {
        const type = block.type;
        const metadata = block.metadata || {};
        const dataSize = block.dataSize || block.size || 0;

        if (type === 'TEXT') {
            const content = this.parser.getTextContent(index);
            return `
                <div class="detail-section">
                    <div class="detail-section-title"> Text Content</div>
                    <div class="content-preview text-content">
                        <pre>${this.escapeHtml(content || 'Unable to read content')}</pre>
                    </div>
                </div>
            `;
        }

        if (type === 'AUDI') {
            return `
                <div class="detail-section">
                    <div class="detail-section-title"> Audio Content</div>
                    <div class="content-preview audio-content">
                        <div class="media-info">
                            <div class="media-icon"></div>
                            <div class="media-details">
                                <div class="media-row"><span class="media-label">Format:</span> ${metadata.format || metadata.codec || 'Unknown'}</div>
                                ${metadata.duration ? `<div class="media-row"><span class="media-label">Duration:</span> ${this.formatDuration(metadata.duration)}</div>` : ''}
                                ${metadata.sample_rate ? `<div class="media-row"><span class="media-label">Sample Rate:</span> ${metadata.sample_rate?.toLocaleString()} Hz</div>` : ''}
                                ${metadata.channels ? `<div class="media-row"><span class="media-label">Channels:</span> ${metadata.channels} (${metadata.channels === 1 ? 'Mono' : metadata.channels === 2 ? 'Stereo' : 'Surround'})</div>` : ''}
                                ${metadata.bitrate ? `<div class="media-row"><span class="media-label">Bitrate:</span> ${Math.round(metadata.bitrate / 1000)} kbps</div>` : ''}
                                ${metadata.bit_depth ? `<div class="media-row"><span class="media-label">Bit Depth:</span> ${metadata.bit_depth}-bit</div>` : ''}
                                <div class="media-row"><span class="media-label">Raw Size:</span> ${MAIFParser.formatSize(dataSize)}</div>
                            </div>
                        </div>
                        <div class="waveform-preview">
                            <div class="waveform">▁▂▃▅▆▇█▇▆▅▃▂▁▂▃▅▆▇█▇▆▅▃▂▁▂▃▅▆▇█▇▆▅▃▂▁</div>
                        </div>
                    </div>
                </div>
            `;
        }

        if (type === 'VIDE') {
            return `
                <div class="detail-section">
                    <div class="detail-section-title"> Video Content</div>
                    <div class="content-preview video-content">
                        <div class="media-info">
                            <div class="video-thumbnail">
                                <div class="play-button">▶</div>
                                <div class="video-label">Video Preview</div>
                            </div>
                            <div class="media-details">
                                <div class="media-row"><span class="media-label">Format:</span> ${metadata.format || metadata.codec || 'Unknown'}</div>
                                ${metadata.width && metadata.height ? `<div class="media-row"><span class="media-label">Resolution:</span> ${metadata.width}×${metadata.height}</div>` : ''}
                                ${metadata.duration ? `<div class="media-row"><span class="media-label">Duration:</span> ${this.formatDuration(metadata.duration)}</div>` : ''}
                                ${metadata.fps || metadata.frame_rate ? `<div class="media-row"><span class="media-label">Frame Rate:</span> ${metadata.fps || metadata.frame_rate} fps</div>` : ''}
                                ${metadata.bitrate ? `<div class="media-row"><span class="media-label">Bitrate:</span> ${Math.round(metadata.bitrate / 1000)} kbps</div>` : ''}
                                ${metadata.audio_codec ? `<div class="media-row"><span class="media-label">Audio:</span> ${metadata.audio_codec}</div>` : ''}
                                ${metadata.frame_count ? `<div class="media-row"><span class="media-label">Frames:</span> ${metadata.frame_count?.toLocaleString()}</div>` : ''}
                                <div class="media-row"><span class="media-label">Raw Size:</span> ${MAIFParser.formatSize(dataSize)}</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        if (type === 'IMAG') {
            return `
                <div class="detail-section">
                    <div class="detail-section-title"> Image Content</div>
                    <div class="content-preview image-content">
                        <div class="media-info">
                            <div class="image-thumbnail">
                                <div class="image-placeholder"></div>
                                <div class="image-label">Image Data</div>
                            </div>
                            <div class="media-details">
                                <div class="media-row"><span class="media-label">Format:</span> ${metadata.format || metadata.mime_type || 'Unknown'}</div>
                                ${metadata.width && metadata.height ? `<div class="media-row"><span class="media-label">Dimensions:</span> ${metadata.width}×${metadata.height} px</div>` : ''}
                                ${metadata.color_mode || metadata.channels ? `<div class="media-row"><span class="media-label">Color Mode:</span> ${metadata.color_mode || (metadata.channels === 3 ? 'RGB' : metadata.channels === 4 ? 'RGBA' : 'Grayscale')}</div>` : ''}
                                ${metadata.bit_depth ? `<div class="media-row"><span class="media-label">Bit Depth:</span> ${metadata.bit_depth}-bit</div>` : ''}
                                ${metadata.dpi ? `<div class="media-row"><span class="media-label">DPI:</span> ${metadata.dpi}</div>` : ''}
                                <div class="media-row"><span class="media-label">Raw Size:</span> ${MAIFParser.formatSize(dataSize)}</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        if (type === 'EMBD') {
            const floatCount = Math.floor(dataSize / 4);
            return `
                <div class="detail-section">
                    <div class="detail-section-title"> Embedding Data</div>
                    <div class="content-preview embeddings-content">
                        <div class="media-info">
                            <div class="media-icon"></div>
                            <div class="media-details">
                                <div class="media-row"><span class="media-label">Dimensions:</span> ${metadata.dimensions || floatCount}</div>
                                <div class="media-row"><span class="media-label">Data Type:</span> ${metadata.dtype || 'float32'}</div>
                                ${metadata.model ? `<div class="media-row"><span class="media-label">Model:</span> ${metadata.model}</div>` : ''}
                                <div class="media-row"><span class="media-label">Raw Size:</span> ${MAIFParser.formatSize(dataSize)}</div>
                            </div>
                        </div>
                        <div class="embedding-visualization">
                            <div class="embedding-bar" style="--value: 75%"></div>
                            <div class="embedding-bar" style="--value: 45%"></div>
                            <div class="embedding-bar" style="--value: 90%"></div>
                            <div class="embedding-bar" style="--value: 30%"></div>
                            <div class="embedding-bar" style="--value: 60%"></div>
                            <div class="embedding-bar" style="--value: 80%"></div>
                            <div class="embedding-bar" style="--value: 25%"></div>
                            <div class="embedding-bar" style="--value: 55%"></div>
                        </div>
                    </div>
                </div>
            `;
        }

        if (type === 'KNOW' || type === 'KGRF') {
            const content = this.parser.getTextContent(index);
            let graphData = null;
            try {
                graphData = JSON.parse(content);
            } catch (e) {}
            
            return `
                <div class="detail-section">
                    <div class="detail-section-title"> Knowledge Graph</div>
                    <div class="content-preview knowledge-content">
                        ${graphData ? `
                            <div class="media-info">
                                <div class="media-icon"></div>
                                <div class="media-details">
                                    <div class="media-row"><span class="media-label">Nodes:</span> ${graphData.nodes?.length || 0}</div>
                                    <div class="media-row"><span class="media-label">Edges:</span> ${graphData.edges?.length || 0}</div>
                                    <div class="media-row"><span class="media-label">Raw Size:</span> ${MAIFParser.formatSize(dataSize)}</div>
                                </div>
                            </div>
                            <pre class="key-display" style="max-height: 200px; overflow: auto;">${JSON.stringify(graphData, null, 2)}</pre>
                        ` : `
                            <pre class="key-display">${this.escapeHtml(content || 'Unable to read content')}</pre>
                        `}
                    </div>
                </div>
            `;
        }

        if (type === 'META' || type === 'BINA') {
            const content = this.parser.getTextContent(index);
            let jsonData = null;
            try {
                jsonData = JSON.parse(content);
            } catch (e) {}
            
            if (jsonData) {
                return `
                    <div class="detail-section">
                        <div class="detail-section-title">${type === 'META' ? ' Metadata Content' : ' Binary Content'}</div>
                        <div class="content-preview json-content">
                            <pre class="key-display">${JSON.stringify(jsonData, null, 2)}</pre>
                        </div>
                    </div>
                `;
            } else {
                // Show hex preview for non-JSON binary
                return this.getHexPreviewHtml(index, dataSize);
            }
        }

        // Default: try to show content or hex dump
        const content = this.parser.getTextContent(index);
        if (content && content.length > 0) {
            // Check if mostly printable
            const printableRatio = (content.match(/[\x20-\x7E\n\r\t]/g) || []).length / content.length;
            if (printableRatio > 0.8) {
                return `
                    <div class="detail-section">
                        <div class="detail-section-title">📄 Content Preview</div>
                        <div class="content-preview text-content">
                            <pre>${this.escapeHtml(content.slice(0, 2000))}${content.length > 2000 ? '\n... (truncated)' : ''}</pre>
                        </div>
                    </div>
                `;
            }
        }
        
        return this.getHexPreviewHtml(index, dataSize);
    }

    getHexPreviewHtml(index, dataSize) {
        const blockData = this.parser.getBlockData(index);
        if (!blockData) return '';
        
        const hexLines = [];
        const maxBytes = Math.min(blockData.length, 256);
        
        for (let i = 0; i < maxBytes; i += 16) {
            const lineBytes = blockData.slice(i, Math.min(i + 16, maxBytes));
            const offset = i.toString(16).padStart(6, '0');
            const hex = Array.from(lineBytes).map(b => b.toString(16).padStart(2, '0')).join(' ').padEnd(47, ' ');
            const ascii = Array.from(lineBytes).map(b => (b >= 32 && b < 127) ? String.fromCharCode(b) : '.').join('');
            hexLines.push(`${offset}  ${hex}  ${ascii}`);
        }
        
        if (blockData.length > 256) {
            hexLines.push(`... ${(blockData.length - 256).toLocaleString()} more bytes ...`);
        }
        
        return `
            <div class="detail-section">
                <div class="detail-section-title"> Binary Content Preview</div>
                <div class="content-preview hex-content">
                    <pre>${hexLines.join('\n')}</pre>
                </div>
            </div>
        `;
    }

    formatDuration(seconds) {
        if (typeof seconds !== 'number') return seconds;
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hrs > 0) {
            return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    escapeHtml(text) {
        if (!text) return '';
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    formatAction(action) {
        const actionMap = {
            'genesis': '🌟 Genesis',
            'add_text_block': ' Add Text',
            'add_embeddings_block': ' Add Embeddings',
            'add_knowledge_graph': ' Add Knowledge Graph',
            'add_image_block': ' Add Image',
            'add_audio_block': ' Add Audio',
            'add_video_block': ' Add Video',
            'update': '✏️ Update',
            'delete': '🗑️ Delete',
            'finalize': ' Finalize',
            'sign': '✍️ Sign',
            'verify': ' Verify'
        };
        return actionMap[action] || action;
    }

    truncateHash(hash, length = 16) {
        if (!hash) return 'N/A';
        if (hash.length <= length * 2) return hash;
        return `${hash.slice(0, length)}...${hash.slice(-length)}`;
    }

    truncateSignature(sig, length = 40) {
        if (!sig) return 'N/A';
        if (sig.length <= length) return sig;
        return `${sig.slice(0, length)}...`;
    }

    showError(message) {
        alert(message); // Simple error display
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.explorer = new MAIFExplorer();
});


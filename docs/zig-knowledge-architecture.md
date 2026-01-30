# Zig Knowledge for Claude Code

A comprehensive guide to building version-aware Zig documentation search with semantic understanding for Claude Code agents.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Architecture Options](#architecture-options)
3. [Recommended Architecture](#recommended-architecture)
4. [Installation Scopes](#installation-scopes)
5. [Database Schema](#database-schema)
6. [MCP Server Tools](#mcp-server-tools)
7. [Skill Content](#skill-content)
8. [ZLS Integration (Optional)](#zls-integration-optional)
9. [Changelog Integration](#changelog-integration)
10. [Version Update Flow](#version-update-flow)
11. [Embedding Model Options](#embedding-model-options)
12. [Quick Start Options](#quick-start-options)
13. [Full Implementation Roadmap](#full-implementation-roadmap)
14. [Resources](#resources)

---

## Problem Statement

Coding agents constantly use the wrong version of Zig and hit compile errors because:

1. **Training data lag**: Models are trained on older Zig versions
2. **Rapid API churn**: Zig's stdlib changes significantly between releases (e.g., "Writergate" in 0.15)
3. **No version awareness**: Agents don't know which version a project targets
4. **No authoritative source**: Agents hallucinate APIs instead of checking docs

**Goal**: Provide Claude Code with:
- Semantic search over version-specific Zig documentation
- Awareness of breaking changes between versions
- Optional ZLS integration for live project feedback
- Distributable as a plugin for the community

---

## Architecture Options

### Option 1: Simple Skill (Static Text)

**Pros**: Zero dependencies, instant loading, no external process  
**Cons**: Limited to what fits in context, no search, manual updates

```
~/.claude/skills/zig/SKILL.md
```

Best for: Quick reference patterns, version gotchas, small snippets.

### Option 2: MCP Server with Embedded Docs + Semantic Search

**Pros**: Full docs searchable, semantic understanding, version-aware  
**Cons**: Requires running process, needs embedding model, more complex

```
zig-knowledge/
├── mcp/server.ts
├── index/zig.sqlite
└── docs/0.15.2/...
```

Best for: Comprehensive documentation access, large codebases.

### Option 3: MCP Server + ZLS Integration

**Pros**: Live project awareness, real-time diagnostics, completions  
**Cons**: Requires ZLS installed, more moving parts

Best for: Active development, catching errors before compile.

### Option 4: Hybrid (Recommended)

**Skill** for instant patterns + **MCP** for deep search + optional **ZLS** for live feedback.

---

## Recommended Architecture

### Directory Structure

```
zig-knowledge/
├── plugin.json                    # Claude Code plugin manifest
├── README.md
├── package.json                   # npm/bun package for distribution
├── install.sh                     # Post-install setup script
│
├── skills/
│   └── zig-patterns/
│       └── SKILL.md               # Version-specific patterns (auto-loaded)
│
├── mcp/
│   ├── index.ts                   # CLI entry point
│   ├── server.ts                  # MCP server implementation
│   ├── search.ts                  # BM25 + vector search
│   ├── embed.ts                   # Embedding generation
│   └── zls.ts                     # Optional ZLS client
│
├── docs/                          # Scraped/converted documentation
│   ├── 0.15.2/
│   │   ├── langref.md             # Language reference
│   │   ├── release-notes.md       # Release notes
│   │   └── stdlib/                # Standard library docs
│   │       ├── std.mem.md
│   │       ├── std.fs.md
│   │       └── ...
│   ├── 0.14.1/
│   │   └── ...
│   └── changelogs/
│       ├── 0.14-to-0.15.md        # Diff summaries
│       └── ...
│
├── index/
│   └── zig.sqlite                 # FTS5 + sqlite-vec database
│
├── scripts/
│   ├── scrape-docs.ts             # Fetch from ziglang.org
│   ├── convert-html.ts            # HTML to markdown
│   ├── extract-symbols.ts         # Parse symbol index
│   ├── generate-changelog.ts      # Diff between versions
│   └── update-version.ts          # Full update pipeline
│
├── mcp-servers/
│   └── config.json                # MCP config for plugin install
│
└── commands/                      # Optional slash commands
    ├── zig-version.md             # /zig-version
    └── zig-check.md               # /zig-check
```

### Plugin Manifest (plugin.json)

```json
{
  "name": "zig-knowledge",
  "version": "0.1.0",
  "description": "Zig documentation with semantic search and version awareness",
  "author": "Your Name",
  "repository": "https://github.com/you/zig-knowledge",
  "license": "MIT",
  "keywords": ["zig", "documentation", "search", "semantic"]
}
```

### Package Configuration (package.json)

```json
{
  "name": "zig-knowledge-mcp",
  "version": "0.1.0",
  "type": "module",
  "bin": {
    "zig-mcp": "./mcp/index.ts"
  },
  "scripts": {
    "serve": "bun run mcp/index.ts serve",
    "index": "bun run scripts/update-version.ts",
    "scrape": "bun run scripts/scrape-docs.ts"
  },
  "dependencies": {
    "@anthropic-ai/sdk": "^0.30.0",
    "better-sqlite3": "^11.0.0",
    "sqlite-vec": "^0.1.0"
  },
  "devDependencies": {
    "bun-types": "latest"
  }
}
```

### MCP Server Config (mcp-servers/config.json)

```json
{
  "zig-docs": {
    "command": "zig-mcp",
    "args": ["serve"],
    "env": {
      "ZIG_VERSION": "0.15.2",
      "ZIG_DOCS_PATH": "${HOME}/.cache/zig-knowledge/index.sqlite"
    }
  }
}
```

---

## Installation Scopes

### User-Level (All Projects)

Location: `~/.claude/`

```
~/.claude/
├── settings.json          # MCP server configs
├── skills/
│   └── zig-knowledge/
│       └── SKILL.md
└── agents/
    └── zig-expert.md      # Optional specialized agent
```

**settings.json**:
```json
{
  "mcpServers": {
    "zig": {
      "command": "zig-mcp",
      "args": ["serve"],
      "env": {
        "ZIG_DOCS_PATH": "~/.cache/zig-knowledge/index.sqlite"
      }
    }
  }
}
```

### Project-Level (Single Repo)

Location: `.claude/` in repo root

```
your-zig-project/
├── .claude/
│   ├── settings.json      # Override version
│   ├── skills/
│   │   └── project-zig/
│   │       └── SKILL.md   # Project-specific patterns
│   └── CLAUDE.md
├── build.zig
├── build.zig.zon
└── src/
```

**Project .claude/settings.json** (overrides user-level):
```json
{
  "mcpServers": {
    "zig": {
      "command": "zig-mcp",
      "args": ["serve"],
      "env": {
        "ZIG_VERSION": "0.14.1"
      }
    }
  }
}
```

### Plugin Installation (For Distribution)

Users install via Claude Code:

```
# From GitHub
/plugins install https://github.com/you/zig-knowledge

# From marketplace (if published)
/plugins marketplace add anthropics/community
/plugins install zig-knowledge
```

### NPX/Bunx (Zero Install)

For one-off use without permanent installation:

```json
{
  "mcpServers": {
    "zig": {
      "command": "npx",
      "args": ["-y", "zig-knowledge-mcp@latest", "serve"]
    }
  }
}
```

---

## Database Schema

Using SQLite with FTS5 for keyword search and sqlite-vec for semantic search.

```sql
-- Main documents table
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,              -- "0.15.2/stdlib/std.mem.md"
    version TEXT NOT NULL,           -- "0.15.2"
    doc_type TEXT NOT NULL,          -- "langref" | "stdlib" | "changelog" | "release-notes"
    title TEXT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,      -- SHA256 for change detection
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_documents_version ON documents(version);
CREATE INDEX idx_documents_type ON documents(doc_type);
CREATE UNIQUE INDEX idx_documents_path ON documents(path);

-- FTS5 full-text search index
CREATE VIRTUAL TABLE documents_fts USING fts5(
    title,
    content,
    content=documents,
    content_rowid=id,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, content)
    VALUES (new.id, new.title, new.content);
END;

CREATE TRIGGER documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, content)
    VALUES ('delete', old.id, old.title, old.content);
END;

CREATE TRIGGER documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, content)
    VALUES ('delete', old.id, old.title, old.content);
    INSERT INTO documents_fts(rowid, title, content)
    VALUES (new.id, new.title, new.content);
END;

-- Vector embeddings for semantic search (chunked)
CREATE TABLE content_vectors (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_seq INTEGER NOT NULL,      -- Chunk number within document
    chunk_start INTEGER NOT NULL,    -- Character offset in original
    chunk_text TEXT NOT NULL,        -- The chunk content (~6KB)
    embedding BLOB,                  -- Raw float32 array
    UNIQUE(doc_id, chunk_seq)
);

CREATE INDEX idx_vectors_doc ON content_vectors(doc_id);

-- sqlite-vec virtual table for ANN search
CREATE VIRTUAL TABLE vectors_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding float[768]             -- Dimension depends on model
);

-- Symbol index for fast API lookups
CREATE TABLE symbols (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    symbol_path TEXT NOT NULL,       -- "std.mem.Allocator.alloc"
    symbol_type TEXT,                -- "fn" | "struct" | "const" | "enum" | "union"
    signature TEXT,                  -- Function signature if applicable
    doc_comment TEXT,                -- Doc comment content
    line_number INTEGER
);

CREATE INDEX idx_symbols_path ON symbols(symbol_path);
CREATE INDEX idx_symbols_version ON symbols(version);
CREATE INDEX idx_symbols_type ON symbols(symbol_type);

-- Changelog entries for version diffs
CREATE TABLE changelog_entries (
    id INTEGER PRIMARY KEY,
    from_version TEXT NOT NULL,
    to_version TEXT NOT NULL,
    category TEXT NOT NULL,          -- "breaking" | "deprecated" | "added" | "fixed"
    title TEXT NOT NULL,
    description TEXT,
    migration_guide TEXT,            -- How to update code
    UNIQUE(from_version, to_version, title)
);

CREATE INDEX idx_changelog_versions ON changelog_entries(from_version, to_version);
CREATE INDEX idx_changelog_category ON changelog_entries(category);

-- Cached embeddings for queries (optional optimization)
CREATE TABLE query_cache (
    query_hash TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## MCP Server Tools

### Tool Definitions

```typescript
const tools = {
  zig_search: {
    name: "zig_search",
    description: "Fast BM25 keyword search across Zig documentation. Best for exact terms, function names, error messages.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search keywords (e.g., 'ArrayList append' or 'error.OutOfMemory')"
        },
        version: {
          type: "string",
          description: "Zig version to search (e.g., '0.15.2'). Defaults to project version."
        },
        doc_type: {
          type: "string",
          enum: ["langref", "stdlib", "changelog", "all"],
          description: "Filter by document type"
        },
        limit: {
          type: "number",
          description: "Max results (default: 5)"
        }
      },
      required: ["query"]
    }
  },

  zig_vsearch: {
    name: "zig_vsearch",
    description: "Semantic vector search across Zig documentation. Best for conceptual queries like 'how to read files' or 'memory allocation patterns'.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Natural language query"
        },
        version: {
          type: "string",
          description: "Zig version to search"
        },
        limit: {
          type: "number",
          description: "Max results (default: 5)"
        }
      },
      required: ["query"]
    }
  },

  zig_query: {
    name: "zig_query",
    description: "Hybrid search combining BM25, semantic search, and LLM reranking. Best quality but slower. Use for complex questions.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search query (keywords or natural language)"
        },
        version: {
          type: "string",
          description: "Zig version to search"
        },
        limit: {
          type: "number",
          description: "Max results (default: 5)"
        }
      },
      required: ["query"]
    }
  },

  zig_symbol: {
    name: "zig_symbol",
    description: "Get documentation for a specific standard library symbol. Fast lookup by exact path.",
    inputSchema: {
      type: "object",
      properties: {
        symbol: {
          type: "string",
          description: "Full symbol path (e.g., 'std.mem.Allocator', 'std.fs.File.reader')"
        },
        version: {
          type: "string",
          description: "Zig version"
        }
      },
      required: ["symbol"]
    }
  },

  zig_changelog: {
    name: "zig_changelog",
    description: "Get breaking changes and migration guide between Zig versions.",
    inputSchema: {
      type: "object",
      properties: {
        from_version: {
          type: "string",
          description: "Source version (e.g., '0.14.1')"
        },
        to_version: {
          type: "string",
          description: "Target version (e.g., '0.15.2')"
        },
        category: {
          type: "string",
          enum: ["breaking", "deprecated", "added", "all"],
          description: "Filter by change category"
        }
      },
      required: ["from_version", "to_version"]
    }
  },

  zig_version: {
    name: "zig_version",
    description: "Get the configured Zig version for the current project. Checks build.zig.zon, environment, or falls back to default.",
    inputSchema: {
      type: "object",
      properties: {},
      required: []
    }
  }
};
```

### Example Tool Responses

**zig_search**:
```json
{
  "results": [
    {
      "path": "0.15.2/stdlib/std.ArrayList.md",
      "title": "std.ArrayList",
      "score": 0.92,
      "snippet": "...pub fn append(self: *Self, item: T) Allocator.Error!void...",
      "doc_type": "stdlib"
    }
  ],
  "query": "ArrayList append",
  "version": "0.15.2",
  "total_results": 12
}
```

**zig_symbol**:
```json
{
  "symbol": "std.fs.File.writer",
  "version": "0.15.2",
  "type": "fn",
  "signature": "pub fn writer(self: File, buffer: []u8) Writer",
  "doc_comment": "Returns a Writer for this file. The buffer is used for buffering writes.",
  "deprecated": false,
  "see_also": ["std.fs.File.Reader", "std.Io.Writer"]
}
```

**zig_changelog**:
```json
{
  "from_version": "0.14.1",
  "to_version": "0.15.2",
  "breaking_changes": [
    {
      "title": "Writergate: std.io.Writer API rewrite",
      "description": "All std.io readers and writers deprecated in favor of std.Io.Reader and std.Io.Writer",
      "migration": "var stdout = std.io.getStdOut().writer();\n→\nvar buf: [4096]u8 = undefined;\nvar w = std.fs.File.stdout().writer(&buf);\nconst stdout = &w.interface;"
    },
    {
      "title": "usingnamespace removed",
      "description": "The usingnamespace keyword has been removed from the language",
      "migration": "Use explicit imports or @fieldParentPtr patterns for mixins"
    }
  ],
  "deprecated": [...],
  "added": [...]
}
```

---

## Skill Content

The skill provides instant context without requiring a tool call. It should contain version-specific patterns and common gotchas.

### skills/zig-patterns/SKILL.md

```markdown
# Zig Development Patterns

Use this skill when writing Zig code. ALWAYS verify the project's Zig version first using `zig_version` tool.

## Version Detection

Check project version in this order:
1. `build.zig.zon` → `.minimum_zig_version` field
2. Environment variable `ZIG_VERSION`
3. Output of `zig version` command
4. Default to latest stable (0.15.2)

## Zig 0.15.x Patterns (Current Stable)

### I/O - Post-Writergate (BREAKING from 0.14)

The entire I/O system was rewritten. Old code WILL NOT COMPILE.

```zig
// ❌ OLD (0.14 and earlier) - DOES NOT WORK
const stdout = std.io.getStdOut().writer();
try stdout.print("Hello\n", .{});

// ✅ NEW (0.15+) - Buffer is now required
var stdout_buffer: [4096]u8 = undefined;
var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
const stdout = &stdout_writer.interface;
try stdout.print("Hello\n", .{});
try stdout.flush();  // DON'T FORGET TO FLUSH!
```

### File Reading (0.15+)

```zig
// ✅ Correct 0.15 pattern
var file = try std.fs.cwd().openFile("data.txt", .{});
defer file.close();

var read_buffer: [8192]u8 = undefined;
var reader = file.reader(&read_buffer);
const content = try reader.readAllAlloc(allocator, max_size);
```

### ArrayList Changes

```zig
// 0.14: std.ArrayList(T)
// 0.15: std.array_list.Managed(T) - but prefer unmanaged

// ✅ Preferred pattern (works in both versions)
var list = std.ArrayListUnmanaged(u8){};
defer list.deinit(allocator);
try list.append(allocator, item);
```

### Format Strings

```zig
// ❌ OLD - ambiguous
std.debug.print("{}", .{my_formattable});

// ✅ NEW - explicit
std.debug.print("{f}", .{my_formattable});  // calls .format()
std.debug.print("{any}", .{my_formattable}); // skips .format()
```

### usingnamespace Removed

```zig
// ❌ REMOVED in 0.15
pub usingnamespace @import("other.zig");

// ✅ Explicit imports
const other = @import("other.zig");
pub const foo = other.foo;
pub const bar = other.bar;

// ✅ For mixins, use @fieldParentPtr pattern
pub const Mixin = struct {
    pub fn method(self: *@This()) void {
        const parent: *Parent = @fieldParentPtr("mixin", self);
        // ...
    }
};
```

## Zig 0.14.x Patterns

If targeting 0.14.x, the old I/O patterns still work:

```zig
// Works in 0.14, NOT in 0.15
const stdout = std.io.getStdOut().writer();
var bw = std.io.bufferedWriter(stdout);
const w = bw.writer();
try w.print("Hello\n", .{});
try bw.flush();
```

## Common Errors and Solutions

### "no field named 'writer' in struct 'fs.File'"
**Cause**: Using 0.14 API on 0.15  
**Fix**: Use `file.reader(&buffer)` or `file.writer(&buffer)` with explicit buffer

### "error: use of undefined value here causes illegal behavior"  
**Cause**: 0.15 stricter undefined handling  
**Fix**: Initialize all values explicitly

### "error: no field named 'root_source_file'"
**Cause**: Build system API change in 0.15  
**Fix**: Use `root_module` field instead

## When Unsure

Use MCP tools:
- `zig_symbol` for exact API lookup
- `zig_search` for keyword search  
- `zig_changelog` for version differences
- `zig_query` for complex questions
```

---

## ZLS Integration (Optional)

For live project awareness, integrate with Zig Language Server.

### ZLS Client (mcp/zls.ts)

```typescript
import { spawn, ChildProcess } from 'child_process';
import { createInterface } from 'readline';

interface LSPMessage {
  jsonrpc: '2.0';
  id?: number;
  method?: string;
  params?: unknown;
  result?: unknown;
  error?: { code: number; message: string };
}

export class ZLSClient {
  private process: ChildProcess | null = null;
  private requestId = 0;
  private pending = new Map<number, { resolve: Function; reject: Function }>();

  async start(workspacePath: string): Promise<void> {
    this.process = spawn('zls', [], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: workspacePath,
    });

    // Handle incoming messages
    const rl = createInterface({ input: this.process.stdout! });
    let buffer = '';
    
    rl.on('line', (line) => {
      if (line.startsWith('Content-Length:')) {
        // Header
      } else if (line === '') {
        // End of headers
      } else {
        buffer += line;
        try {
          const msg: LSPMessage = JSON.parse(buffer);
          this.handleMessage(msg);
          buffer = '';
        } catch {
          // Incomplete JSON, continue buffering
        }
      }
    });

    // Initialize
    await this.request('initialize', {
      processId: process.pid,
      capabilities: {},
      rootUri: `file://${workspacePath}`,
    });
    
    await this.notify('initialized', {});
  }

  async hover(file: string, line: number, col: number): Promise<string | null> {
    const result = await this.request('textDocument/hover', {
      textDocument: { uri: `file://${file}` },
      position: { line: line - 1, character: col - 1 },
    });
    return result?.contents?.value ?? null;
  }

  async diagnostics(file: string): Promise<Array<{
    line: number;
    message: string;
    severity: string;
  }>> {
    // ZLS pushes diagnostics, so we need to track them
    // This is a simplified version
    return [];
  }

  async completions(file: string, line: number, col: number): Promise<string[]> {
    const result = await this.request('textDocument/completion', {
      textDocument: { uri: `file://${file}` },
      position: { line: line - 1, character: col - 1 },
    });
    return (result?.items ?? []).map((i: any) => i.label);
  }

  private async request(method: string, params: unknown): Promise<any> {
    const id = ++this.requestId;
    const msg = { jsonrpc: '2.0', id, method, params };
    
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.send(msg);
    });
  }

  private notify(method: string, params: unknown): void {
    this.send({ jsonrpc: '2.0', method, params });
  }

  private send(msg: object): void {
    const json = JSON.stringify(msg);
    const header = `Content-Length: ${Buffer.byteLength(json)}\r\n\r\n`;
    this.process?.stdin?.write(header + json);
  }

  private handleMessage(msg: LSPMessage): void {
    if (msg.id !== undefined && this.pending.has(msg.id)) {
      const { resolve, reject } = this.pending.get(msg.id)!;
      this.pending.delete(msg.id);
      if (msg.error) {
        reject(new Error(msg.error.message));
      } else {
        resolve(msg.result);
      }
    }
  }

  async stop(): Promise<void> {
    await this.request('shutdown', {});
    this.notify('exit', {});
    this.process?.kill();
  }
}
```

### Additional MCP Tools for ZLS

```typescript
const zlsTools = {
  zls_hover: {
    name: "zls_hover",
    description: "Get type info and documentation at a specific position in a Zig file",
    inputSchema: {
      type: "object",
      properties: {
        file: { type: "string", description: "Path to .zig file" },
        line: { type: "number", description: "Line number (1-indexed)" },
        col: { type: "number", description: "Column number (1-indexed)" }
      },
      required: ["file", "line", "col"]
    }
  },

  zls_diagnostics: {
    name: "zls_diagnostics",
    description: "Get current compile errors and warnings from ZLS",
    inputSchema: {
      type: "object",
      properties: {
        file: { type: "string", description: "Path to .zig file (optional, all files if omitted)" }
      },
      required: []
    }
  },

  zls_completions: {
    name: "zls_completions",
    description: "Get valid completions at a position",
    inputSchema: {
      type: "object",
      properties: {
        file: { type: "string", description: "Path to .zig file" },
        line: { type: "number", description: "Line number (1-indexed)" },
        col: { type: "number", description: "Column number (1-indexed)" }
      },
      required: ["file", "line", "col"]
    }
  }
};
```

---

## Changelog Integration

### Scraping Release Notes

The release notes at `https://ziglang.org/download/X.Y.Z/release-notes.html` contain:
- Breaking changes with migration guides
- New features
- Bug fixes
- Standard library changes

### Changelog Extraction Script

```typescript
// scripts/generate-changelog.ts
import { JSDOM } from 'jsdom';

interface ChangelogEntry {
  category: 'breaking' | 'deprecated' | 'added' | 'fixed';
  title: string;
  description: string;
  migration?: string;
  codeOld?: string;
  codeNew?: string;
}

async function extractChangelog(
  fromVersion: string,
  toVersion: string
): Promise<ChangelogEntry[]> {
  const url = `https://ziglang.org/download/${toVersion}/release-notes.html`;
  const response = await fetch(url);
  const html = await response.text();
  const dom = new JSDOM(html);
  const doc = dom.window.document;
  
  const entries: ChangelogEntry[] = [];
  
  // Find sections with breaking changes
  const sections = doc.querySelectorAll('h3, h4');
  
  for (const section of sections) {
    const title = section.textContent?.trim() ?? '';
    const content = [];
    let sibling = section.nextElementSibling;
    
    while (sibling && !['H2', 'H3', 'H4'].includes(sibling.tagName)) {
      content.push(sibling.textContent);
      sibling = sibling.nextElementSibling;
    }
    
    // Categorize based on keywords
    const text = content.join(' ').toLowerCase();
    let category: ChangelogEntry['category'] = 'added';
    
    if (text.includes('removed') || text.includes('breaking') || text.includes('deleted')) {
      category = 'breaking';
    } else if (text.includes('deprecated')) {
      category = 'deprecated';
    } else if (text.includes('fixed') || text.includes('bug')) {
      category = 'fixed';
    }
    
    // Extract code examples
    const codeBlocks = section.parentElement?.querySelectorAll('pre code') ?? [];
    const codes = Array.from(codeBlocks).map(c => c.textContent ?? '');
    
    entries.push({
      category,
      title,
      description: content.join('\n').slice(0, 500),
      codeOld: codes[0],
      codeNew: codes[1],
    });
  }
  
  return entries;
}
```

### Key 0.14 → 0.15 Breaking Changes

Pre-extracted for the skill:

| Change | Category | Impact |
|--------|----------|--------|
| Writergate I/O rewrite | Breaking | All I/O code must be updated |
| `usingnamespace` removed | Breaking | Must use explicit imports |
| `async`/`await` keywords removed | Breaking | Use std.Io interface instead |
| `std.ArrayList` → `std.array_list.Managed` | Deprecated | Prefer `ArrayListUnmanaged` |
| `BoundedArray` removed | Breaking | Use `ArrayListUnmanaged` with buffer |
| Format string `{f}` required | Breaking | Explicit format method calls |
| `std.fifo.LinearFifo` removed | Breaking | Use `std.Io.Reader`/`Writer` |
| Build system `root_source_file` removed | Breaking | Use `root_module` |

---

## Version Update Flow

When a new Zig version releases:

```bash
# 1. Run the update script
./scripts/update-version.ts 0.16.0

# What it does:
# - Downloads https://ziglang.org/documentation/0.16.0/ 
# - Downloads https://ziglang.org/download/0.16.0/release-notes.html
# - Converts HTML to markdown
# - Extracts symbol index from stdlib docs
# - Generates changelog diff from 0.15.2
# - Inserts into SQLite
# - Generates embeddings via Ollama
# - Updates SKILL.md with new patterns
```

### Update Script

```typescript
// scripts/update-version.ts
import { scrapeLanguageRef, scrapeStdlib, scrapeReleaseNotes } from './scrape-docs';
import { convertToMarkdown } from './convert-html';
import { extractSymbols } from './extract-symbols';
import { generateChangelog } from './generate-changelog';
import { indexDocuments, generateEmbeddings } from '../mcp/embed';
import { updateSkill } from './update-skill';

async function updateVersion(version: string) {
  console.log(`Updating Zig docs for version ${version}...`);
  
  // 1. Scrape documentation
  console.log('Scraping language reference...');
  const langrefHtml = await scrapeLanguageRef(version);
  
  console.log('Scraping stdlib docs...');
  const stdlibHtml = await scrapeStdlib(version);
  
  console.log('Scraping release notes...');
  const releaseNotesHtml = await scrapeReleaseNotes(version);
  
  // 2. Convert to markdown
  console.log('Converting to markdown...');
  const docs = {
    langref: convertToMarkdown(langrefHtml),
    stdlib: stdlibHtml.map(convertToMarkdown),
    releaseNotes: convertToMarkdown(releaseNotesHtml),
  };
  
  // 3. Extract symbols
  console.log('Extracting symbols...');
  const symbols = extractSymbols(docs.stdlib, version);
  
  // 4. Generate changelog
  const previousVersion = getPreviousVersion(version);
  console.log(`Generating changelog from ${previousVersion}...`);
  const changelog = await generateChangelog(previousVersion, version);
  
  // 5. Index in database
  console.log('Indexing documents...');
  await indexDocuments(docs, version);
  await indexSymbols(symbols);
  await indexChangelog(changelog, previousVersion, version);
  
  // 6. Generate embeddings
  console.log('Generating embeddings (this may take a while)...');
  await generateEmbeddings(version);
  
  // 7. Update skill with new patterns
  console.log('Updating skill...');
  await updateSkill(version, changelog);
  
  console.log(`✓ Version ${version} indexed successfully!`);
}

// Run
const version = process.argv[2];
if (!version) {
  console.error('Usage: update-version.ts <version>');
  process.exit(1);
}
updateVersion(version);
```

---

## Embedding Model Options

### Option 1: Ollama (Recommended for Local)

```typescript
const OLLAMA_URL = process.env.OLLAMA_URL ?? 'http://localhost:11434';

// Models (auto-pulled if missing)
const EMBED_MODEL = 'nomic-embed-text';      // ~274MB, 768 dims
const RERANK_MODEL = 'qwen3-reranker:0.6b';  // ~640MB
const EXPAND_MODEL = 'qwen3:0.6b';           // ~400MB

async function embed(texts: string[]): Promise<number[][]> {
  const response = await fetch(`${OLLAMA_URL}/api/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: EMBED_MODEL,
      input: texts,
    }),
  });
  const data = await response.json();
  return data.embeddings;
}
```

### Option 2: OpenAI-Compatible API

```typescript
const OPENAI_URL = process.env.OPENAI_API_URL ?? 'https://api.openai.com/v1';
const OPENAI_KEY = process.env.OPENAI_API_KEY;

async function embed(texts: string[]): Promise<number[][]> {
  const response = await fetch(`${OPENAI_URL}/embeddings`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${OPENAI_KEY}`,
    },
    body: JSON.stringify({
      model: 'text-embedding-3-small',
      input: texts,
    }),
  });
  const data = await response.json();
  return data.data.map((d: any) => d.embedding);
}
```

### Option 3: Local ONNX (No Network)

```typescript
import { pipeline } from '@xenova/transformers';

let embedder: any = null;

async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return embedder;
}

async function embed(texts: string[]): Promise<number[][]> {
  const model = await getEmbedder();
  const results = await Promise.all(
    texts.map(async (text) => {
      const output = await model(text, { pooling: 'mean', normalize: true });
      return Array.from(output.data);
    })
  );
  return results;
}
```

### Chunking Strategy

Following qmd's approach (~6KB chunks):

```typescript
const CHUNK_SIZE = 6000;  // Characters, roughly fits in token window
const CHUNK_OVERLAP = 200;

function chunkDocument(content: string, title: string): Array<{
  text: string;
  seq: number;
  start: number;
}> {
  const chunks: Array<{ text: string; seq: number; start: number }> = [];
  let pos = 0;
  let seq = 0;
  
  while (pos < content.length) {
    const end = Math.min(pos + CHUNK_SIZE, content.length);
    
    // Try to break at paragraph boundary
    let breakPoint = end;
    if (end < content.length) {
      const lastPara = content.lastIndexOf('\n\n', end);
      if (lastPara > pos + CHUNK_SIZE / 2) {
        breakPoint = lastPara;
      }
    }
    
    const chunkText = content.slice(pos, breakPoint);
    
    // Format for embedding: "title | text"
    chunks.push({
      text: `${title} | ${chunkText}`,
      seq,
      start: pos,
    });
    
    pos = breakPoint - CHUNK_OVERLAP;
    seq++;
  }
  
  return chunks;
}
```

---

## Quick Start Options

### Option A: qmd + Skill (5 Minutes)

Use Tobi's qmd for search, add a skill for patterns:

```bash
# 1. Install qmd
bun install -g https://github.com/tobi/qmd

# 2. Download and index Zig docs
mkdir -p ~/zig-docs/0.15.2
# (manually save docs as .md files or use a scraper)

cd ~/zig-docs
qmd add .
qmd add-context ./0.15.2 "Zig 0.15.2 documentation"
qmd embed

# 3. Add MCP config
cat >> ~/.claude/settings.json << 'EOF'
{
  "mcpServers": {
    "qmd": {
      "command": "qmd",
      "args": ["mcp"]
    }
  }
}
EOF

# 4. Add skill
mkdir -p ~/.claude/skills/zig
# Copy the SKILL.md content from above
```

### Option B: Skill Only (2 Minutes)

Just the patterns, no search:

```bash
mkdir -p ~/.claude/skills/zig-patterns
cat > ~/.claude/skills/zig-patterns/SKILL.md << 'EOF'
# Zig 0.15.x Patterns
[paste skill content here]
EOF
```

### Option C: Full Implementation (1-2 Hours)

Build the complete MCP server:

```bash
# 1. Create repo
mkdir zig-knowledge && cd zig-knowledge
bun init

# 2. Install dependencies
bun add better-sqlite3 sqlite-vec

# 3. Create structure
mkdir -p mcp scripts docs index skills/zig-patterns

# 4. Implement (see Full Implementation Roadmap)
```

---

## Full Implementation Roadmap

### Phase 1: Core Infrastructure (MVP)

- [ ] SQLite schema setup
- [ ] Basic MCP server with stdio transport
- [ ] `zig_version` tool (reads build.zig.zon)
- [ ] `zig_search` tool (BM25 only)
- [ ] Manual doc import (copy markdown files)
- [ ] Basic skill with 0.15 patterns

### Phase 2: Documentation Pipeline

- [ ] HTML scraper for ziglang.org
- [ ] HTML to markdown converter
- [ ] Symbol extractor for stdlib
- [ ] Automated version update script
- [ ] Changelog generator

### Phase 3: Semantic Search

- [ ] Ollama integration for embeddings
- [ ] sqlite-vec setup
- [ ] `zig_vsearch` tool
- [ ] Chunking pipeline
- [ ] Query caching

### Phase 4: Advanced Features

- [ ] `zig_query` with reranking
- [ ] `zig_symbol` fast lookup
- [ ] `zig_changelog` version diff
- [ ] ZLS integration (optional)

### Phase 5: Distribution

- [ ] Plugin manifest
- [ ] Install script
- [ ] NPM/Bun package
- [ ] Documentation
- [ ] GitHub Actions for auto-updates

---

## Resources

### Zig Documentation

- Language Reference: `https://ziglang.org/documentation/{version}/`
- Stdlib Docs: `https://ziglang.org/documentation/{version}/std/`
- Release Notes: `https://ziglang.org/download/{version}/release-notes.html`
- Download Page: `https://ziglang.org/download/`
- JSON Index: `https://ziglang.org/download/index.json`

### Claude Code

- Docs: https://code.claude.com/docs
- Skills: https://code.claude.com/docs/en/skills  
- Subagents: https://code.claude.com/docs/en/sub-agents
- MCP: https://code.claude.com/docs/en/mcp
- Plugins: https://code.claude.com/docs/en/plugins
- Hooks: https://code.claude.com/docs/en/hooks

### Reference Implementations

- qmd (Tobi's search): https://github.com/tobi/qmd
- awesome-claude-code: https://github.com/hesreallyhim/awesome-claude-code
- MCP specification: https://modelcontextprotocol.io

### Embedding Models

- nomic-embed-text: https://ollama.com/library/nomic-embed-text
- embeddinggemma: https://ollama.com/library/embeddinggemma
- Xenova/transformers: https://github.com/xenova/transformers.js

---

## Notes

- Zig 0.15.2 released 2025-10-11 (current stable)
- Zig 0.16.0 in development (master)
- Major breaking changes in 0.15: Writergate, usingnamespace removal
- ZLS version should match Zig version

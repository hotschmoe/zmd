# Deep Dive: Porting QMD to Zig

## Executive Summary

QMD is a local hybrid search engine for markdown documents created by Tobi Lütke (Shopify CEO). It combines BM25 full-text search, vector semantic search, and LLM re-ranking—all running locally. The TypeScript/Bun implementation relies on SQLite (FTS5), sqlite-vec, and node-llama-cpp for inference.

A Zig port offers compelling advantages: single static binary distribution, superior memory control for embedding operations, potential for custom SIMD-optimized vector operations, and elimination of the Bun/Node runtime dependency. However, the LLM inference layer presents the primary challenge.

---

## 1. Architecture Analysis

### Current Stack (TypeScript/Bun)

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                            │
│                    (Bun + src/qmd.ts)                       │
├─────────────────────────────────────────────────────────────┤
│                    MCP Server Layer                         │
│              (@modelcontextprotocol/sdk)                    │
├─────────────────────────────────────────────────────────────┤
│                   Search Pipeline                           │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────┐  │
│  │  BM25   │  │  Vector  │  │   RRF   │  │  LLM Rerank  │  │
│  │  FTS5   │  │  Search  │  │ Fusion  │  │              │  │
│  └─────────┘  └──────────┘  └─────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   Storage Layer                             │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │   SQLite + FTS5     │  │      sqlite-vec             │  │
│  │   (documents_fts)   │  │  (vectors_vec, cosine sim)  │  │
│  └─────────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   LLM Layer                                 │
│                 (node-llama-cpp)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │ Embeddings  │  │  Reranker   │  │ Query Expansion  │   │
│  │ embeddingge │  │ qwen3-reran │  │    qwen3:0.6b    │   │
│  └─────────────┘  └─────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Target Stack (Zig)

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                            │
│              (std.process, std.heap, clap?)                 │
├─────────────────────────────────────────────────────────────┤
│                    MCP Server Layer                         │
│        (Custom JSON-RPC over stdio implementation)          │
├─────────────────────────────────────────────────────────────┤
│                   Search Pipeline                           │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────┐  │
│  │  BM25   │  │  Vector  │  │   RRF   │  │  LLM Rerank  │  │
│  │  FTS5   │  │  SIMD    │  │ Fusion  │  │              │  │
│  └─────────┘  └──────────┘  └─────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   Storage Layer                             │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │   sqlite-zig        │  │  Native Zig vec storage     │  │
│  │   (with FTS5)       │  │  or sqlite-vec C bindings   │  │
│  └─────────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   LLM Layer                                 │
│              (llama.cpp C API bindings)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │ Embeddings  │  │  Reranker   │  │ Query Expansion  │   │
│  │ GGUF models │  │ GGUF models │  │   GGUF models    │   │
│  └─────────────┘  └─────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component-by-Component Porting Analysis

### 2.1 SQLite + FTS5 (Difficulty: Low-Medium)

**Current Implementation:**
- Bun's built-in SQLite driver
- FTS5 virtual table for BM25 scoring
- Schema: `collections`, `path_contexts`, `documents`, `documents_fts`

**Zig Options:**

1. **sqlite-zig** (https://github.com/vrischmann/zig-sqlite)
   - Mature bindings to SQLite C library
   - FTS5 support depends on SQLite compile flags
   - Recommended approach

2. **Direct C interop** with SQLite amalgamation
   - Include sqlite3.c directly in build
   - Full control over compile flags (ensure `-DSQLITE_ENABLE_FTS5`)
   - More work but maximum flexibility

**FTS5 Schema Translation:**

```sql
-- These queries work identically in Zig via sqlite-zig
CREATE VIRTUAL TABLE documents_fts USING fts5(
    title, 
    content,
    tokenize="porter unicode61"
);

-- BM25 ranking query
SELECT 
    d.*, 
    bm25(documents_fts) as score
FROM documents_fts 
JOIN documents d ON documents_fts.rowid = d.rowid
WHERE documents_fts MATCH ?
ORDER BY score
LIMIT ?;
```

**Key Implementation Notes:**
- SQLite's FTS5 BM25 returns negative scores (lower = better match)
- QMD normalizes with `Math.abs(score)`
- Porter stemmer is built into FTS5 tokenizer

### 2.2 Vector Storage & Search (Difficulty: Medium)

**Current Implementation:**
- sqlite-vec extension for vector similarity
- Stores 768-dim float vectors (embeddinggemma output)
- Cosine distance similarity search

**Zig Options:**

1. **Native Zig Vector Operations** (Recommended for your style)
   ```zig
   const Vector = struct {
       data: []f32,
       
       pub fn cosineSimilarity(self: Vector, other: Vector) f32 {
           var dot: f32 = 0;
           var normA: f32 = 0;
           var normB: f32 = 0;
           
           // SIMD-friendly loop
           for (self.data, other.data) |a, b| {
               dot += a * b;
               normA += a * a;
               normB += b * b;
           }
           
           return dot / (@sqrt(normA) * @sqrt(normB));
       }
   };
   ```

2. **sqlite-vec C bindings**
   - Load as SQLite extension
   - Less control but simpler integration

3. **Custom SIMD Implementation**
   ```zig
   const std = @import("std");
   
   pub fn cosineSimilaritySIMD(a: []const f32, b: []const f32) f32 {
       const vec_len = std.simd.suggestVectorLength(f32) orelse 4;
       const Vec = @Vector(vec_len, f32);
       
       var dot_sum: Vec = @splat(0);
       var norm_a: Vec = @splat(0);
       var norm_b: Vec = @splat(0);
       
       var i: usize = 0;
       while (i + vec_len <= a.len) : (i += vec_len) {
           const va: Vec = a[i..][0..vec_len].*;
           const vb: Vec = b[i..][0..vec_len].*;
           
           dot_sum += va * vb;
           norm_a += va * va;
           norm_b += vb * vb;
       }
       
       // Reduce and handle remainder...
       const dot = @reduce(.Add, dot_sum);
       const na = @sqrt(@reduce(.Add, norm_a));
       const nb = @sqrt(@reduce(.Add, norm_b));
       
       return dot / (na * nb);
   }
   ```

**Vector Storage Schema:**
```zig
const EmbeddingChunk = struct {
    hash: [6]u8,      // Document content hash (first 6 chars)
    seq: u16,         // Chunk sequence number
    pos: u32,         // Character position in original
    embedding: [768]f32,  // 768-dim vector
};
```

### 2.3 LLM Inference Layer (Difficulty: HIGH - Primary Challenge)

**Current Implementation (node-llama-cpp):**
- Downloads GGUF models from HuggingFace
- Three model types:
  1. **Embedding**: `embeddinggemma-300M-Q8_0.gguf` (768-dim output)
  2. **Reranker**: `qwen3-reranker-0.6b-q8_0.gguf` (cross-encoder)
  3. **Generation**: `qmd-query-expansion-1.7B-q4_k_m.gguf` (query expansion)

**Zig Options:**

#### Option A: llama.cpp C API Bindings (Recommended)

llama.cpp exposes a clean C API that Zig can bind to directly:

```zig
const c = @cImport({
    @cInclude("llama.h");
});

const LlamaModel = struct {
    model: *c.llama_model,
    ctx: *c.llama_context,
    
    pub fn init(model_path: [:0]const u8) !LlamaModel {
        const model_params = c.llama_model_default_params();
        const model = c.llama_load_model_from_file(model_path, model_params) 
            orelse return error.ModelLoadFailed;
        
        var ctx_params = c.llama_context_default_params();
        ctx_params.n_ctx = 2048;
        ctx_params.n_batch = 512;
        
        const ctx = c.llama_new_context_with_model(model, ctx_params)
            orelse return error.ContextCreateFailed;
        
        return .{ .model = model, .ctx = ctx };
    }
    
    pub fn embed(self: *LlamaModel, text: []const u8) ![768]f32 {
        // Tokenize input
        var tokens: [512]c.llama_token = undefined;
        const n_tokens = c.llama_tokenize(
            self.model,
            text.ptr,
            @intCast(text.len),
            &tokens,
            tokens.len,
            true,  // add_bos
            false, // special
        );
        
        // Run inference for embeddings
        c.llama_decode(self.ctx, c.llama_batch_get_one(&tokens, n_tokens, 0, 0));
        
        // Extract embedding from hidden state
        var embedding: [768]f32 = undefined;
        // ... extraction logic depends on model architecture
        
        return embedding;
    }
    
    pub fn deinit(self: *LlamaModel) void {
        c.llama_free(self.ctx);
        c.llama_free_model(self.model);
    }
};
```

**Build Integration:**
```zig
// build.zig
const llama = b.dependency("llama_cpp", .{});
exe.linkLibrary(llama.artifact("llama"));
exe.addIncludePath(llama.path("include"));
```

#### Option B: ggml Direct Integration

For maximum control (especially for Laminae integration):

```zig
const ggml = @cImport({
    @cInclude("ggml.h");
    @cInclude("ggml-backend.h");
});

// Lower-level tensor operations
// More work but could integrate with CEVA architecture
```

#### Option C: External Process (Fallback)

Shell out to `llama-cli` or similar:
```zig
const result = try std.process.Child.run(.{
    .allocator = allocator,
    .argv = &[_][]const u8{
        "llama-embedding",
        "-m", model_path,
        "--prompt", text,
    },
});
```
- Simpler but slower, loses memory efficiency gains

### 2.4 Reciprocal Rank Fusion (Difficulty: Low)

Pure algorithmic translation—this is straightforward:

```zig
const RRFResult = struct {
    doc_id: u64,
    score: f32,
};

pub fn reciprocalRankFusion(
    allocator: std.mem.Allocator,
    result_lists: []const []const RRFResult,
    k: f32,  // Usually 60
    weights: []const f32,
) ![]RRFResult {
    var scores = std.AutoHashMap(u64, f32).init(allocator);
    defer scores.deinit();
    
    for (result_lists, 0..) |list, list_idx| {
        const weight = if (list_idx < weights.len) weights[list_idx] else 1.0;
        
        for (list, 0..) |result, rank| {
            const rrf_score = weight / (k + @as(f32, @floatFromInt(rank)) + 1.0);
            
            const entry = try scores.getOrPut(result.doc_id);
            if (entry.found_existing) {
                entry.value_ptr.* += rrf_score;
            } else {
                entry.value_ptr.* = rrf_score;
            }
        }
    }
    
    // Sort by score descending
    var items = try allocator.alloc(RRFResult, scores.count());
    var i: usize = 0;
    var iter = scores.iterator();
    while (iter.next()) |entry| {
        items[i] = .{ .doc_id = entry.key_ptr.*, .score = entry.value_ptr.* };
        i += 1;
    }
    
    std.sort.pdq(RRFResult, items, {}, struct {
        fn lessThan(_: void, a: RRFResult, b: RRFResult) bool {
            return b.score < a.score;  // Descending
        }
    }.lessThan);
    
    return items;
}
```

### 2.5 MCP Server (Difficulty: Medium)

MCP (Model Context Protocol) is JSON-RPC 2.0 over stdio. Native Zig implementation:

```zig
const std = @import("std");
const json = std.json;

const MCPServer = struct {
    allocator: std.mem.Allocator,
    tools: std.StringHashMap(ToolHandler),
    
    const ToolHandler = *const fn (params: json.Value) anyerror!json.Value;
    
    pub fn init(allocator: std.mem.Allocator) MCPServer {
        return .{
            .allocator = allocator,
            .tools = std.StringHashMap(ToolHandler).init(allocator),
        };
    }
    
    pub fn registerTool(self: *MCPServer, name: []const u8, handler: ToolHandler) !void {
        try self.tools.put(name, handler);
    }
    
    pub fn run(self: *MCPServer) !void {
        const stdin = std.io.getStdIn().reader();
        const stdout = std.io.getStdOut().writer();
        
        var buf: [64 * 1024]u8 = undefined;
        
        while (true) {
            const line = stdin.readUntilDelimiter(&buf, '\n') catch |err| {
                if (err == error.EndOfStream) break;
                return err;
            };
            
            const request = try json.parseFromSlice(json.Value, self.allocator, line, .{});
            defer request.deinit();
            
            const response = try self.handleRequest(request.value);
            try json.stringify(response, .{}, stdout);
            try stdout.writeByte('\n');
        }
    }
    
    fn handleRequest(self: *MCPServer, request: json.Value) !json.Value {
        const method = request.object.get("method").?.string;
        const params = request.object.get("params") orelse .null;
        const id = request.object.get("id");
        
        if (std.mem.eql(u8, method, "tools/call")) {
            const tool_name = params.object.get("name").?.string;
            const tool_params = params.object.get("arguments") orelse .null;
            
            if (self.tools.get(tool_name)) |handler| {
                const result = try handler(tool_params);
                return self.successResponse(id, result);
            }
            return self.errorResponse(id, -32601, "Tool not found");
        }
        
        // Handle other MCP methods: initialize, tools/list, etc.
        return self.errorResponse(id, -32601, "Method not found");
    }
    
    fn successResponse(self: *MCPServer, id: ?json.Value, result: json.Value) json.Value {
        _ = self;
        var obj = json.ObjectMap.init(self.allocator);
        obj.put("jsonrpc", .{ .string = "2.0" }) catch unreachable;
        obj.put("id", id orelse .null) catch unreachable;
        obj.put("result", result) catch unreachable;
        return .{ .object = obj };
    }
    
    fn errorResponse(self: *MCPServer, id: ?json.Value, code: i32, message: []const u8) json.Value {
        _ = self;
        // Build error response...
    }
};
```

### 2.6 CLI & Argument Parsing (Difficulty: Low)

```zig
const std = @import("std");
const clap = @import("clap");  // Or roll your own

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const params = comptime clap.parseParamsComptime(
        \\-h, --help        Display help
        \\-n, --num <NUM>   Number of results (default: 5)
        \\--min-score <F>   Minimum score threshold
        \\--full            Show full document content
        \\--json            JSON output
        \\--files           Output: score,filepath,context
        \\<COMMAND>         Command to run
        \\<ARGS>...         Command arguments
    );
    
    var args = try clap.parse(params, std.process.args(), allocator);
    defer args.deinit();
    
    const command = args.positionals[0];
    
    if (std.mem.eql(u8, command, "search")) {
        try runSearch(allocator, args);
    } else if (std.mem.eql(u8, command, "vsearch")) {
        try runVectorSearch(allocator, args);
    } else if (std.mem.eql(u8, command, "query")) {
        try runHybridQuery(allocator, args);
    } else if (std.mem.eql(u8, command, "mcp")) {
        try runMCPServer(allocator);
    }
    // ... etc
}
```

### 2.7 Markdown Parsing (Difficulty: Low-Medium)

QMD extracts titles from markdown (first heading). Simple implementation:

```zig
const MarkdownDoc = struct {
    title: ?[]const u8,
    content: []const u8,
    
    pub fn parse(allocator: std.mem.Allocator, raw: []const u8) !MarkdownDoc {
        var title: ?[]const u8 = null;
        
        // Find first heading
        var lines = std.mem.splitSequence(u8, raw, "\n");
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t");
            
            // ATX heading: # Title
            if (std.mem.startsWith(u8, trimmed, "#")) {
                var start: usize = 0;
                while (start < trimmed.len and trimmed[start] == '#') : (start += 1) {}
                title = std.mem.trim(u8, trimmed[start..], " \t");
                break;
            }
            
            // Setext heading: check for underline on next line
            if (lines.peek()) |next_line| {
                const next_trimmed = std.mem.trim(u8, next_line, " \t");
                if (next_trimmed.len > 0 and 
                    (std.mem.allEqual(u8, next_trimmed, '=') or 
                     std.mem.allEqual(u8, next_trimmed, '-'))) {
                    title = try allocator.dupe(u8, trimmed);
                    break;
                }
            }
        }
        
        return .{
            .title = title,
            .content = raw,
        };
    }
};
```

---

## 3. Data Flow Implementation

### Search Pipeline in Zig

```zig
const SearchPipeline = struct {
    db: *Database,
    embedder: *LlamaModel,
    reranker: *LlamaModel,
    expander: *LlamaModel,
    
    pub fn hybridQuery(
        self: *SearchPipeline,
        allocator: std.mem.Allocator,
        query: []const u8,
        limit: usize,
    ) ![]SearchResult {
        // 1. Query Expansion
        const expanded_queries = try self.expandQuery(allocator, query);
        defer allocator.free(expanded_queries);
        
        // 2. Run BM25 + Vector search for each query variant
        var all_results = std.ArrayList(RRFInput).init(allocator);
        defer all_results.deinit();
        
        // Original query with 2x weight
        try all_results.append(.{
            .bm25 = try self.db.ftsSearch(query, 30),
            .vector = try self.vectorSearch(query, 30),
            .weight = 2.0,
        });
        
        // Expanded queries
        for (expanded_queries) |eq| {
            try all_results.append(.{
                .bm25 = try self.db.ftsSearch(eq, 30),
                .vector = try self.vectorSearch(eq, 30),
                .weight = 1.0,
            });
        }
        
        // 3. RRF Fusion
        const fused = try reciprocalRankFusion(allocator, all_results.items, 60);
        
        // 4. Take top 30 for reranking
        const candidates = fused[0..@min(30, fused.len)];
        
        // 5. LLM Reranking
        const reranked = try self.rerankDocuments(allocator, query, candidates);
        
        // 6. Position-aware blending
        const blended = try self.positionAwareBlend(allocator, fused, reranked);
        
        return blended[0..@min(limit, blended.len)];
    }
    
    fn vectorSearch(self: *SearchPipeline, query: []const u8, limit: usize) ![]SearchResult {
        // Format query for embeddinggemma
        const formatted = try std.fmt.allocPrint(
            self.allocator,
            "task: search result | query: {s}",
            .{query},
        );
        defer self.allocator.free(formatted);
        
        const query_embedding = try self.embedder.embed(formatted);
        return self.db.vectorSimilaritySearch(query_embedding, limit);
    }
    
    fn expandQuery(self: *SearchPipeline, allocator: std.mem.Allocator, query: []const u8) ![][]const u8 {
        const prompt = try std.fmt.allocPrint(
            allocator,
            \\Generate 2 alternative search queries for: "{s}"
            \\Return only the queries, one per line.
        ,
            .{query},
        );
        defer allocator.free(prompt);
        
        const response = try self.expander.generate(prompt, 150);
        
        // Parse response into lines
        var queries = std.ArrayList([]const u8).init(allocator);
        var lines = std.mem.splitSequence(u8, response, "\n");
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len > 0) {
                try queries.append(try allocator.dupe(u8, trimmed));
            }
        }
        
        return queries.toOwnedSlice();
    }
    
    fn rerankDocuments(
        self: *SearchPipeline,
        allocator: std.mem.Allocator,
        query: []const u8,
        candidates: []const SearchResult,
    ) ![]RerankResult {
        var results = try allocator.alloc(RerankResult, candidates.len);
        
        for (candidates, 0..) |candidate, i| {
            const doc_content = try self.db.getDocumentContent(candidate.doc_id);
            
            const prompt = try std.fmt.allocPrint(allocator,
                \\<Instruct>: Given a search query, determine if the document is relevant.
                \\<Query>: {s}
                \\<Document>: {s}
            , .{ query, doc_content[0..@min(2000, doc_content.len)] });
            defer allocator.free(prompt);
            
            // Get yes/no with logprobs
            const response = try self.reranker.generateWithLogprobs(prompt, 1);
            
            results[i] = .{
                .doc_id = candidate.doc_id,
                .score = response.confidence,  // P("yes")
            };
        }
        
        return results;
    }
    
    fn positionAwareBlend(
        self: *SearchPipeline,
        allocator: std.mem.Allocator,
        rrf_results: []const SearchResult,
        rerank_results: []const RerankResult,
    ) ![]SearchResult {
        _ = self;
        var blended = try allocator.alloc(SearchResult, rrf_results.len);
        
        for (rrf_results, 0..) |rrf, rank| {
            // Find matching rerank result
            const rerank_score = for (rerank_results) |rr| {
                if (rr.doc_id == rrf.doc_id) break rr.score;
            } else 0.5;
            
            // Position-aware weighting
            const rrf_weight: f32 = if (rank < 3) 0.75 
                else if (rank < 10) 0.60 
                else 0.40;
            
            blended[rank] = .{
                .doc_id = rrf.doc_id,
                .score = rrf_weight * rrf.score + (1.0 - rrf_weight) * rerank_score,
            };
        }
        
        // Re-sort by blended score
        std.sort.pdq(SearchResult, blended, {}, struct {
            fn cmp(_: void, a: SearchResult, b: SearchResult) bool {
                return b.score < a.score;
            }
        }.cmp);
        
        return blended;
    }
};
```

---

## 4. Build System Configuration

```zig
// build.zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    
    const exe = b.addExecutable(.{
        .name = "qmd",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // SQLite with FTS5
    exe.addCSourceFile(.{
        .file = b.path("vendor/sqlite3.c"),
        .flags = &.{
            "-DSQLITE_ENABLE_FTS5",
            "-DSQLITE_ENABLE_JSON1",
            "-DSQLITE_DQS=0",
        },
    });
    exe.addIncludePath(b.path("vendor"));
    
    // llama.cpp
    const llama_dep = b.dependency("llama_cpp", .{
        .target = target,
        .optimize = optimize,
    });
    exe.linkLibrary(llama_dep.artifact("llama"));
    exe.addIncludePath(llama_dep.path("include"));
    
    // Link system libraries
    exe.linkLibC();
    if (target.result.os.tag == .macos) {
        exe.linkFramework("Accelerate");
        exe.linkFramework("Metal");
        exe.linkFramework("MetalKit");
    }
    
    b.installArtifact(exe);
    
    // Run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    
    const run_step = b.step("run", "Run qmd");
    run_step.dependOn(&run_cmd.step);
}
```

---

## 5. Model Loading & Caching Strategy

```zig
const ModelManager = struct {
    cache_dir: []const u8,
    models: std.StringHashMap(*LlamaModel),
    allocator: std.mem.Allocator,
    
    const ModelSpec = struct {
        name: []const u8,
        hf_repo: []const u8,
        hf_file: []const u8,
    };
    
    const DEFAULT_MODELS = [_]ModelSpec{
        .{
            .name = "embed",
            .hf_repo = "ggml-org/embeddinggemma-300M-GGUF",
            .hf_file = "embeddinggemma-300M-Q8_0.gguf",
        },
        .{
            .name = "rerank",
            .hf_repo = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
            .hf_file = "qwen3-reranker-0.6b-q8_0.gguf",
        },
        .{
            .name = "expand",
            .hf_repo = "tobil/qmd-query-expansion-1.7B-gguf",
            .hf_file = "qmd-query-expansion-1.7B-q4_k_m.gguf",
        },
    };
    
    pub fn getModel(self: *ModelManager, name: []const u8) !*LlamaModel {
        if (self.models.get(name)) |model| {
            return model;
        }
        
        // Find spec and load
        for (DEFAULT_MODELS) |spec| {
            if (std.mem.eql(u8, spec.name, name)) {
                const path = try self.ensureModelDownloaded(spec);
                const model = try LlamaModel.init(path);
                try self.models.put(name, model);
                return model;
            }
        }
        
        return error.UnknownModel;
    }
    
    fn ensureModelDownloaded(self: *ModelManager, spec: ModelSpec) ![:0]const u8 {
        const path = try std.fs.path.join(self.allocator, &.{
            self.cache_dir,
            "models",
            spec.hf_file,
        });
        
        // Check if exists
        std.fs.accessAbsolute(path, .{}) catch {
            // Download from HuggingFace
            try self.downloadModel(spec, path);
        };
        
        return try self.allocator.dupeZ(u8, path);
    }
    
    fn downloadModel(self: *ModelManager, spec: ModelSpec, dest: []const u8) !void {
        const url = try std.fmt.allocPrint(
            self.allocator,
            "https://huggingface.co/{s}/resolve/main/{s}",
            .{ spec.hf_repo, spec.hf_file },
        );
        defer self.allocator.free(url);
        
        // Use std.http.Client for download
        // ... implementation
    }
};
```

---

## 6. Chunking Strategy

QMD uses token-based chunking (800 tokens, 15% overlap):

```zig
const Chunker = struct {
    tokenizer: *Tokenizer,  // From llama.cpp
    chunk_tokens: usize = 800,
    overlap_pct: f32 = 0.15,
    
    const Chunk = struct {
        text: []const u8,
        seq: u16,
        start_pos: u32,
        end_pos: u32,
    };
    
    pub fn chunkDocument(self: *Chunker, allocator: std.mem.Allocator, text: []const u8) ![]Chunk {
        const tokens = try self.tokenizer.tokenize(text);
        defer allocator.free(tokens);
        
        const overlap_tokens = @as(usize, @intFromFloat(
            @as(f32, @floatFromInt(self.chunk_tokens)) * self.overlap_pct
        ));
        const stride = self.chunk_tokens - overlap_tokens;
        
        var chunks = std.ArrayList(Chunk).init(allocator);
        
        var i: usize = 0;
        var seq: u16 = 0;
        while (i < tokens.len) {
            const end = @min(i + self.chunk_tokens, tokens.len);
            const chunk_tokens = tokens[i..end];
            
            const chunk_text = try self.tokenizer.detokenize(allocator, chunk_tokens);
            const start_pos = try self.findTextPosition(text, chunk_text);
            
            try chunks.append(.{
                .text = chunk_text,
                .seq = seq,
                .start_pos = @intCast(start_pos),
                .end_pos = @intCast(start_pos + chunk_text.len),
            });
            
            seq += 1;
            i += stride;
        }
        
        return chunks.toOwnedSlice();
    }
};
```

---

## 7. Implementation Roadmap

### Phase 1: Core Foundation (2-3 weeks)
- [ ] SQLite + FTS5 integration with zig-sqlite
- [ ] Basic CLI structure
- [ ] Document indexing (`collection add`)
- [ ] BM25 search (`search` command)
- [ ] Basic output formatting

### Phase 2: Vector Search (2 weeks)
- [ ] llama.cpp bindings for embeddings
- [ ] Vector storage in SQLite (BLOB or custom table)
- [ ] SIMD-optimized cosine similarity
- [ ] `vsearch` command
- [ ] Token-based chunking

### Phase 3: Hybrid Search (2 weeks)
- [ ] RRF fusion implementation
- [ ] Query expansion via LLM
- [ ] LLM reranking with logprobs
- [ ] Position-aware blending
- [ ] `query` command

### Phase 4: MCP & Polish (1-2 weeks)
- [ ] JSON-RPC MCP server
- [ ] Model auto-download from HuggingFace
- [ ] Output formats (JSON, CSV, MD, XML)
- [ ] Context management
- [ ] Multi-index support

### Phase 5: Optimization (Ongoing)
- [ ] Memory pooling for vector operations
- [ ] Batch embedding generation
- [ ] Parallel search execution
- [ ] Model quantization options

---

## 8. Potential Synergies with Laminae

Given your work on Laminae's CEVA architecture, several interesting synergies emerge:

1. **Vector Storage**: The embedded container addressing could enable interesting distributed vector index designs where chunks "live" in containers

2. **Model Isolation**: Different LLM models (embed, rerank, expand) could run in isolated CEVA containers with their own memory spaces

3. **Parallel Search**: CEVA's addressing could enable elegant parallel BM25 + vector search across document partitions

4. **Memory Efficiency**: Direct control over embedding memory layout without runtime GC pressure

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| llama.cpp API changes | Medium | High | Pin to specific version, abstract behind interface |
| FTS5 tokenizer differences | Low | Medium | Test extensively, document edge cases |
| Model download failures | Medium | Medium | Implement retry logic, checksum validation |
| SIMD portability | Medium | Low | Fallback scalar implementations |
| MCP spec evolution | Low | Medium | Version negotiation in protocol |

---

## 10. Conclusion

Porting QMD to Zig is highly feasible and offers genuine benefits: a single ~5MB static binary (plus model files), no runtime dependencies, superior memory control for vector operations, and potential integration with your broader Laminae ecosystem.

The primary complexity lies in the llama.cpp integration for LLM inference. I recommend starting with Option A (C API bindings) as it's well-documented and node-llama-cpp itself uses this approach. The rest of the stack—SQLite, FTS5, vector ops, RRF—translates cleanly to idiomatic Zig.

Estimated total effort: 6-10 weeks for feature parity, assuming familiarity with llama.cpp internals.

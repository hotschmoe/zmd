const std = @import("std");
const Database = @import("../storage/database.zig").Database;
const SearchResult = @import("../storage/database.zig").SearchResult;
const VectorSearch = @import("vector.zig").VectorSearch;
const SimilarityResult = @import("vector.zig").SimilarityResult;
const RRF = @import("rrf.zig").RRF;
const FusedResult = @import("rrf.zig").FusedResult;
const RankedResult = @import("rrf.zig").RankedResult;
const ResultList = @import("rrf.zig").ResultList;
const LLM = @import("../llm/engine.zig").LLM;

/// Unified search pipeline implementing QMD's hybrid search strategy
pub const SearchPipeline = struct {
    allocator: std.mem.Allocator,
    db: *Database,
    vector_search: VectorSearch,
    rrf: RRF,
    llm: ?*LLM,

    pub fn init(allocator: std.mem.Allocator, db: *Database, llm: ?*LLM) SearchPipeline {
        return .{
            .allocator = allocator,
            .db = db,
            .vector_search = VectorSearch.init(allocator, db),
            .rrf = RRF.init(allocator),
            .llm = llm,
        };
    }

    pub fn deinit(self: *SearchPipeline) void {
        self.vector_search.deinit();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Simple BM25 Search
    // ═══════════════════════════════════════════════════════════════════════

    pub fn search(
        self: *SearchPipeline,
        query: []const u8,
        options: SearchOptions,
    ) ![]HybridResult {
        const fts_results = try self.db.ftsSearch(self.allocator, query, options.limit);

        // Convert to HybridResult format
        var results = try self.allocator.alloc(HybridResult, fts_results.len);
        for (fts_results, 0..) |r, i| {
            results[i] = .{
                .doc_id = r.doc_id,
                .path = r.path,
                .title = r.title,
                .collection = r.collection,
                .score = normalizeScore(r.score, .bm25),
                .snippet = r.snippet,
                .source = .bm25,
            };
        }

        // Filter by min_score
        if (options.min_score > 0) {
            results = try self.filterByScore(results, options.min_score);
        }

        return results;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Vector Semantic Search
    // ═══════════════════════════════════════════════════════════════════════

    pub fn vsearch(
        self: *SearchPipeline,
        query: []const u8,
        options: SearchOptions,
    ) ![]HybridResult {
        // Get query embedding
        const llm = self.llm orelse return error.LLMNotInitialized;

        // Format query for embeddinggemma
        const formatted_query = try std.fmt.allocPrint(
            self.allocator,
            "task: search result | query: {s}",
            .{query},
        );
        defer self.allocator.free(formatted_query);

        const query_embedding = try llm.embed(formatted_query);

        // Ensure vectors are loaded
        if (self.vector_search.vectors == null) {
            try self.vector_search.loadVectors();
        }

        const vec_results = try self.vector_search.search(
            query_embedding,
            options.limit,
            options.min_score,
        );

        // Convert to HybridResult format
        var results = try self.allocator.alloc(HybridResult, vec_results.len);
        for (vec_results, 0..) |r, i| {
            results[i] = .{
                .doc_id = r.doc_id,
                .path = r.path,
                .title = r.title,
                .collection = "", // TODO: get from DB
                .score = r.score,
                .snippet = null,
                .source = .vector,
            };
        }

        return results;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Hybrid Query (Full Pipeline)
    // ═══════════════════════════════════════════════════════════════════════

    pub fn hybridQuery(
        self: *SearchPipeline,
        query: []const u8,
        options: HybridOptions,
    ) ![]HybridResult {
        const llm = self.llm orelse return error.LLMNotInitialized;

        // ─────────────────────────────────────────────────────────────────────
        // Step 1: Query Expansion
        // ─────────────────────────────────────────────────────────────────────

        var all_queries = std.ArrayList(QueryVariant).init(self.allocator);
        defer all_queries.deinit();

        // Original query with 2x weight
        try all_queries.append(.{ .text = query, .weight = 2.0, .is_original = true });

        // Generate expanded queries if enabled
        if (options.enable_query_expansion) {
            const expanded = try llm.expandQuery(query);
            for (expanded) |eq| {
                try all_queries.append(.{ .text = eq, .weight = 1.0, .is_original = false });
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Step 2: Parallel Search (BM25 + Vector for each query)
        // ─────────────────────────────────────────────────────────────────────

        var result_lists = std.ArrayList(ResultList).init(self.allocator);
        defer result_lists.deinit();

        // Ensure vectors are loaded
        if (self.vector_search.vectors == null) {
            try self.vector_search.loadVectors();
        }

        for (all_queries.items) |qv| {
            // BM25 search
            const bm25_results = try self.db.ftsSearch(self.allocator, qv.text, options.candidates_per_query);
            try result_lists.append(.{
                .results = try self.toRankedResults(bm25_results),
                .weight = qv.weight,
            });

            // Vector search
            const formatted = try std.fmt.allocPrint(
                self.allocator,
                "task: search result | query: {s}",
                .{qv.text},
            );
            defer self.allocator.free(formatted);

            const embedding = try llm.embed(formatted);
            const vec_results = try self.vector_search.search(
                embedding,
                options.candidates_per_query,
                0.0,
            );
            try result_lists.append(.{
                .results = try self.vecToRankedResults(vec_results),
                .weight = qv.weight,
            });
        }

        // ─────────────────────────────────────────────────────────────────────
        // Step 3: RRF Fusion
        // ─────────────────────────────────────────────────────────────────────

        const fused = try self.rrf.fuse(result_lists.items, .{
            .k = 60.0,
            .enable_top_rank_bonus = true,
            .rank1_bonus = 0.05,
            .rank2_3_bonus = 0.02,
        });

        // Take top N candidates for reranking
        const candidates_count = @min(options.rerank_candidates, fused.len);
        const candidates = fused[0..candidates_count];

        // ─────────────────────────────────────────────────────────────────────
        // Step 4: LLM Reranking
        // ─────────────────────────────────────────────────────────────────────

        var rerank_scores = std.AutoHashMap(i64, f32).init(self.allocator);
        defer rerank_scores.deinit();

        if (options.enable_reranking) {
            for (candidates) |candidate| {
                // Get document content (truncated)
                const doc_content = try self.getDocumentContent(candidate.doc_id, 2000);
                defer self.allocator.free(doc_content);

                // Rerank using LLM
                const rerank_score = try llm.rerank(query, doc_content);
                try rerank_scores.put(candidate.doc_id, rerank_score);
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Step 5: Position-Aware Blending
        // ─────────────────────────────────────────────────────────────────────

        const blended = if (options.enable_reranking)
            try self.rrf.positionAwareBlend(fused, rerank_scores, .{
                .top3_rrf_weight = 0.75,
                .top10_rrf_weight = 0.60,
                .rest_rrf_weight = 0.40,
            })
        else
            fused;

        // ─────────────────────────────────────────────────────────────────────
        // Step 6: Convert to Final Results
        // ─────────────────────────────────────────────────────────────────────

        const result_count = @min(options.limit, blended.len);
        var results = try self.allocator.alloc(HybridResult, result_count);

        for (blended[0..result_count], 0..) |r, i| {
            results[i] = .{
                .doc_id = r.doc_id,
                .path = r.path,
                .title = r.title,
                .collection = "", // TODO
                .score = r.score,
                .snippet = r.snippet,
                .source = .hybrid,
            };
        }

        // Filter by min_score
        if (options.min_score > 0) {
            results = try self.filterByScore(results, options.min_score);
        }

        return results;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════

    fn toRankedResults(self: *SearchPipeline, search_results: []const SearchResult) ![]const RankedResult {
        var results = try self.allocator.alloc(RankedResult, search_results.len);
        for (search_results, 0..) |r, i| {
            results[i] = .{
                .doc_id = r.doc_id,
                .path = r.path,
                .title = r.title,
                .snippet = r.snippet,
            };
        }
        return results;
    }

    fn vecToRankedResults(self: *SearchPipeline, vec_results: []const SimilarityResult) ![]const RankedResult {
        var results = try self.allocator.alloc(RankedResult, vec_results.len);
        for (vec_results, 0..) |r, i| {
            results[i] = .{
                .doc_id = r.doc_id,
                .path = r.path,
                .title = r.title,
                .snippet = null,
            };
        }
        return results;
    }

    fn getDocumentContent(self: *SearchPipeline, doc_id: i64, max_len: usize) ![]const u8 {
        // TODO: Implement actual document fetch from database
        _ = doc_id;
        _ = max_len;
        _ = self;
        return "";
    }

    fn filterByScore(self: *SearchPipeline, results: []HybridResult, min_score: f32) ![]HybridResult {
        var filtered = std.ArrayList(HybridResult).init(self.allocator);
        for (results) |r| {
            if (r.score >= min_score) {
                try filtered.append(r);
            }
        }
        self.allocator.free(results);
        return filtered.toOwnedSlice();
    }

    fn normalizeScore(score: f32, source: ScoreSource) f32 {
        return switch (source) {
            .bm25 => {
                // BM25 scores are typically 0-25+, normalize to 0-1
                // Using sigmoid-like normalization
                return score / (score + 10.0);
            },
            .vector => score, // Already 0-1
            .hybrid => score, // Already blended
            .reranker => score, // Already 0-1
        };
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

pub const SearchOptions = struct {
    limit: usize = 5,
    min_score: f32 = 0.0,
};

pub const HybridOptions = struct {
    limit: usize = 5,
    min_score: f32 = 0.0,
    candidates_per_query: usize = 30,
    rerank_candidates: usize = 30,
    enable_query_expansion: bool = true,
    enable_reranking: bool = true,
};

pub const ScoreSource = enum {
    bm25,
    vector,
    hybrid,
    reranker,
};

pub const HybridResult = struct {
    doc_id: i64,
    path: []const u8,
    title: ?[]const u8,
    collection: []const u8,
    score: f32,
    snippet: ?[]const u8,
    source: ScoreSource,
};

const QueryVariant = struct {
    text: []const u8,
    weight: f32,
    is_original: bool,
};

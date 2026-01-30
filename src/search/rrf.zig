const std = @import("std");

/// Reciprocal Rank Fusion (RRF) implementation
/// Combines multiple ranked lists into a single fused ranking
pub const RRF = struct {
    allocator: std.mem.Allocator,

    /// Default k parameter (controls ranking smoothness)
    pub const DEFAULT_K: f32 = 60.0;

    pub fn init(allocator: std.mem.Allocator) RRF {
        return .{ .allocator = allocator };
    }

    /// Fuse multiple result lists using RRF algorithm
    /// 
    /// The algorithm computes: score(d) = Σ (weight_i / (k + rank_i(d)))
    /// for each document d across all input lists.
    /// 
    /// Optionally applies top-rank bonuses for documents that appear at
    /// the top of any individual list.
    pub fn fuse(
        self: *RRF,
        result_lists: []const ResultList,
        options: FuseOptions,
    ) ![]FusedResult {
        var scores = std.AutoHashMap(i64, ScoreAccumulator).init(self.allocator);
        defer scores.deinit();

        // Process each result list
        for (result_lists) |list| {
            for (list.results, 0..) |result, rank| {
                // RRF formula: 1 / (k + rank + 1)
                const rrf_score = list.weight / (options.k + @as(f32, @floatFromInt(rank)) + 1.0);

                // Top-rank bonus
                var bonus: f32 = 0;
                if (options.enable_top_rank_bonus) {
                    if (rank == 0) {
                        bonus = options.rank1_bonus;
                    } else if (rank <= 2) {
                        bonus = options.rank2_3_bonus;
                    }
                }

                const entry = try scores.getOrPut(result.doc_id);
                if (entry.found_existing) {
                    entry.value_ptr.score += rrf_score + bonus;
                    if (rank == 0) {
                        entry.value_ptr.top_rank_count += 1;
                    }
                } else {
                    entry.value_ptr.* = .{
                        .score = rrf_score + bonus,
                        .path = result.path,
                        .title = result.title,
                        .snippet = result.snippet,
                        .top_rank_count = if (rank == 0) @as(u32, 1) else 0,
                    };
                }
            }
        }

        // Convert to array and sort
        var results = try self.allocator.alloc(FusedResult, scores.count());
        var i: usize = 0;
        var iter = scores.iterator();
        while (iter.next()) |entry| {
            results[i] = .{
                .doc_id = entry.key_ptr.*,
                .score = entry.value_ptr.score,
                .path = entry.value_ptr.path,
                .title = entry.value_ptr.title,
                .snippet = entry.value_ptr.snippet,
                .top_rank_count = entry.value_ptr.top_rank_count,
            };
            i += 1;
        }

        // Sort by score descending
        std.sort.pdq(FusedResult, results, {}, struct {
            fn cmp(_: void, a: FusedResult, b: FusedResult) bool {
                return b.score < a.score;
            }
        }.cmp);

        return results;
    }

    /// Blend RRF scores with reranker scores using position-aware weighting
    /// 
    /// QMD's position-aware blending:
    /// - Rank 1-3:  75% retrieval, 25% reranker (preserves exact matches)
    /// - Rank 4-10: 60% retrieval, 40% reranker  
    /// - Rank 11+:  40% retrieval, 60% reranker (trust reranker more)
    pub fn positionAwareBlend(
        self: *RRF,
        fused_results: []const FusedResult,
        rerank_scores: std.AutoHashMap(i64, f32),
        options: BlendOptions,
    ) ![]FusedResult {
        var blended = try self.allocator.alloc(FusedResult, fused_results.len);

        for (fused_results, 0..) |result, rank| {
            // Get rerank score (default to 0.5 if not reranked)
            const rerank_score = rerank_scores.get(result.doc_id) orelse 0.5;

            // Determine weight based on rank position
            const rrf_weight: f32 = if (rank < 3)
                options.top3_rrf_weight
            else if (rank < 10)
                options.top10_rrf_weight
            else
                options.rest_rrf_weight;

            // Normalize RRF score to 0-1 range (approximate)
            const max_rrf = 1.0 / (options.k + 1.0);
            const normalized_rrf = @min(result.score / max_rrf, 1.0);

            // Blend scores
            const blended_score = rrf_weight * normalized_rrf + (1.0 - rrf_weight) * rerank_score;

            blended[rank] = .{
                .doc_id = result.doc_id,
                .score = blended_score,
                .path = result.path,
                .title = result.title,
                .snippet = result.snippet,
                .top_rank_count = result.top_rank_count,
            };
        }

        // Re-sort by blended score
        std.sort.pdq(FusedResult, blended, {}, struct {
            fn cmp(_: void, a: FusedResult, b: FusedResult) bool {
                return b.score < a.score;
            }
        }.cmp);

        return blended;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

pub const ResultList = struct {
    results: []const RankedResult,
    weight: f32 = 1.0,
};

pub const RankedResult = struct {
    doc_id: i64,
    path: []const u8,
    title: ?[]const u8,
    snippet: ?[]const u8,
};

pub const FusedResult = struct {
    doc_id: i64,
    score: f32,
    path: []const u8,
    title: ?[]const u8,
    snippet: ?[]const u8,
    top_rank_count: u32,
};

const ScoreAccumulator = struct {
    score: f32,
    path: []const u8,
    title: ?[]const u8,
    snippet: ?[]const u8,
    top_rank_count: u32,
};

pub const FuseOptions = struct {
    k: f32 = RRF.DEFAULT_K,
    enable_top_rank_bonus: bool = true,
    rank1_bonus: f32 = 0.05,
    rank2_3_bonus: f32 = 0.02,
};

pub const BlendOptions = struct {
    k: f32 = RRF.DEFAULT_K,
    top3_rrf_weight: f32 = 0.75,
    top10_rrf_weight: f32 = 0.60,
    rest_rrf_weight: f32 = 0.40,
};

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

test "RRF basic fusion" {
    var rrf = RRF.init(std.testing.allocator);

    const list1 = [_]RankedResult{
        .{ .doc_id = 1, .path = "a.md", .title = null, .snippet = null },
        .{ .doc_id = 2, .path = "b.md", .title = null, .snippet = null },
        .{ .doc_id = 3, .path = "c.md", .title = null, .snippet = null },
    };

    const list2 = [_]RankedResult{
        .{ .doc_id = 2, .path = "b.md", .title = null, .snippet = null },
        .{ .doc_id = 1, .path = "a.md", .title = null, .snippet = null },
        .{ .doc_id = 4, .path = "d.md", .title = null, .snippet = null },
    };

    const results = try rrf.fuse(&[_]ResultList{
        .{ .results = &list1, .weight = 1.0 },
        .{ .results = &list2, .weight = 1.0 },
    }, .{});
    defer std.testing.allocator.free(results);

    // Doc 2 appears first in list2 and second in list1, should have highest score
    // Doc 1 appears first in list1 and second in list2, should be close
    try std.testing.expect(results.len == 4);
    try std.testing.expect(results[0].doc_id == 1 or results[0].doc_id == 2);
}

test "RRF with weights" {
    var rrf = RRF.init(std.testing.allocator);

    const list1 = [_]RankedResult{
        .{ .doc_id = 1, .path = "a.md", .title = null, .snippet = null },
    };

    const list2 = [_]RankedResult{
        .{ .doc_id = 2, .path = "b.md", .title = null, .snippet = null },
    };

    // Give list1 double weight
    const results = try rrf.fuse(&[_]ResultList{
        .{ .results = &list1, .weight = 2.0 },
        .{ .results = &list2, .weight = 1.0 },
    }, .{});
    defer std.testing.allocator.free(results);

    // Doc 1 should have higher score due to 2x weight
    try std.testing.expect(results[0].doc_id == 1);
    try std.testing.expect(results[0].score > results[1].score);
}

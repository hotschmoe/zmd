const std = @import("std");
const Database = @import("../storage/database.zig").Database;
const VectorRecord = @import("../storage/database.zig").VectorRecord;

/// SIMD-optimized vector similarity search
pub const VectorSearch = struct {
    allocator: std.mem.Allocator,
    db: *Database,

    // Cached vectors for fast search
    vectors: ?[]VectorRecord = null,

    pub fn init(allocator: std.mem.Allocator, db: *Database) VectorSearch {
        return .{
            .allocator = allocator,
            .db = db,
        };
    }

    pub fn deinit(self: *VectorSearch) void {
        if (self.vectors) |vecs| {
            for (vecs) |v| {
                self.allocator.free(v.embedding);
                self.allocator.free(v.path);
                if (v.title) |t| self.allocator.free(t);
            }
            self.allocator.free(vecs);
        }
    }

    /// Load all vectors into memory for fast search
    pub fn loadVectors(self: *VectorSearch) !void {
        self.vectors = try self.db.getAllVectors(self.allocator);
    }

    /// Search for similar documents using cosine similarity
    pub fn search(
        self: *VectorSearch,
        query_embedding: []const f32,
        limit: usize,
        min_score: f32,
    ) ![]SimilarityResult {
        const vectors = self.vectors orelse return error.VectorsNotLoaded;

        var results = std.ArrayList(SimilarityResult).init(self.allocator);
        defer results.deinit();

        // Calculate similarity for all vectors
        for (vectors) |vec| {
            const similarity = cosineSimilaritySIMD(query_embedding, vec.embedding);

            if (similarity >= min_score) {
                try results.append(.{
                    .doc_id = vec.doc_id,
                    .chunk_seq = vec.seq,
                    .path = vec.path,
                    .title = vec.title,
                    .score = similarity,
                });
            }
        }

        // Sort by similarity descending
        std.sort.pdq(SimilarityResult, results.items, {}, struct {
            fn cmp(_: void, a: SimilarityResult, b: SimilarityResult) bool {
                return b.score < a.score;
            }
        }.cmp);

        // Return top N
        const result_count = @min(limit, results.items.len);
        const owned = try self.allocator.alloc(SimilarityResult, result_count);
        @memcpy(owned, results.items[0..result_count]);

        return owned;
    }
};

pub const SimilarityResult = struct {
    doc_id: i64,
    chunk_seq: u16,
    path: []const u8,
    title: ?[]const u8,
    score: f32,
};

// ═══════════════════════════════════════════════════════════════════════════
// SIMD-Optimized Cosine Similarity
// ═══════════════════════════════════════════════════════════════════════════

/// Compute cosine similarity using SIMD when available
pub fn cosineSimilaritySIMD(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    // Try to use SIMD vector operations
    const vec_len = std.simd.suggestVectorLength(f32) orelse 4;

    return switch (vec_len) {
        4 => cosineSimilaritySIMDImpl(4, a, b),
        8 => cosineSimilaritySIMDImpl(8, a, b),
        16 => cosineSimilaritySIMDImpl(16, a, b),
        else => cosineSimilarityScalar(a, b),
    };
}

fn cosineSimilaritySIMDImpl(comptime vec_len: comptime_int, a: []const f32, b: []const f32) f32 {
    const Vec = @Vector(vec_len, f32);

    var dot_acc: Vec = @splat(0);
    var norm_a_acc: Vec = @splat(0);
    var norm_b_acc: Vec = @splat(0);

    var i: usize = 0;

    // Process vec_len elements at a time
    while (i + vec_len <= a.len) : (i += vec_len) {
        const va: Vec = a[i..][0..vec_len].*;
        const vb: Vec = b[i..][0..vec_len].*;

        dot_acc += va * vb;
        norm_a_acc += va * va;
        norm_b_acc += vb * vb;
    }

    // Reduce SIMD accumulators
    var dot = @reduce(.Add, dot_acc);
    var norm_a = @reduce(.Add, norm_a_acc);
    var norm_b = @reduce(.Add, norm_b_acc);

    // Handle remainder with scalar ops
    while (i < a.len) : (i += 1) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 0;

    return dot / denom;
}

/// Scalar fallback for cosine similarity
fn cosineSimilarityScalar(a: []const f32, b: []const f32) f32 {
    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    for (a, b) |va, vb| {
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 0;

    return dot / denom;
}

// ═══════════════════════════════════════════════════════════════════════════
// Distance Conversions
// ═══════════════════════════════════════════════════════════════════════════

/// Convert cosine distance (from sqlite-vec) to similarity score
/// sqlite-vec returns distance, we want similarity
pub fn distanceToSimilarity(distance: f32) f32 {
    // cosine_distance = 1 - cosine_similarity
    // so: cosine_similarity = 1 - distance
    // But sqlite-vec uses: 1 / (1 + distance) for some operations
    return 1.0 / (1.0 + distance);
}

/// Normalize a vector in-place
pub fn normalizeVector(vec: []f32) void {
    var norm: f32 = 0;
    for (vec) |v| {
        norm += v * v;
    }
    norm = @sqrt(norm);

    if (norm > 0) {
        for (vec) |*v| {
            v.* /= norm;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

test "cosine similarity - identical vectors" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    const sim = cosineSimilaritySIMD(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sim, 0.0001);
}

test "cosine similarity - orthogonal vectors" {
    const a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0, 0.0, 0.0 };

    const sim = cosineSimilaritySIMD(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sim, 0.0001);
}

test "cosine similarity - opposite vectors" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ -1.0, -2.0, -3.0, -4.0 };

    const sim = cosineSimilaritySIMD(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), sim, 0.0001);
}

test "cosine similarity - large vectors" {
    var a: [768]f32 = undefined;
    var b: [768]f32 = undefined;

    // Fill with predictable values
    for (0..768) |i| {
        a[i] = @floatFromInt(i % 10);
        b[i] = @floatFromInt((i + 5) % 10);
    }

    const sim = cosineSimilaritySIMD(&a, &b);

    // Should be positive but not 1
    try std.testing.expect(sim > 0.5);
    try std.testing.expect(sim < 1.0);
}

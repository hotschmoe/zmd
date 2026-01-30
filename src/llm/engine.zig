const std = @import("std");

// ═══════════════════════════════════════════════════════════════════════════
// llama.cpp C API bindings (conditional compilation)
// ═══════════════════════════════════════════════════════════════════════════

// When llama.cpp is available, uncomment this:
// const c = @cImport({
//     @cInclude("llama.h");
// });

/// LLM inference engine wrapping llama.cpp
/// Provides embedding, reranking, and text generation capabilities
pub const LLM = struct {
    allocator: std.mem.Allocator,
    models: Models,
    cache_dir: []const u8,

    const Models = struct {
        embed: ?*Model = null,
        rerank: ?*Model = null,
        expand: ?*Model = null,
    };

    /// Model specifications (HuggingFace URIs)
    pub const ModelSpec = struct {
        name: []const u8,
        hf_repo: []const u8,
        hf_file: []const u8,
        n_ctx: u32 = 2048,
        n_batch: u32 = 512,
    };

    pub const DEFAULT_MODELS = struct {
        pub const EMBED = ModelSpec{
            .name = "embed",
            .hf_repo = "ggml-org/embeddinggemma-300M-GGUF",
            .hf_file = "embeddinggemma-300M-Q8_0.gguf",
            .n_ctx = 2048,
        };

        pub const RERANK = ModelSpec{
            .name = "rerank",
            .hf_repo = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
            .hf_file = "qwen3-reranker-0.6b-q8_0.gguf",
            .n_ctx = 2048,
        };

        pub const EXPAND = ModelSpec{
            .name = "expand",
            .hf_repo = "tobil/qmd-query-expansion-1.7B-gguf",
            .hf_file = "qmd-query-expansion-1.7B-q4_k_m.gguf",
            .n_ctx = 2048,
        };
    };

    pub fn init(allocator: std.mem.Allocator, cache_dir: []const u8) !LLM {
        // Initialize llama.cpp backend
        // c.llama_backend_init();

        return .{
            .allocator = allocator,
            .models = .{},
            .cache_dir = cache_dir,
        };
    }

    pub fn deinit(self: *LLM) void {
        if (self.models.embed) |m| m.deinit();
        if (self.models.rerank) |m| m.deinit();
        if (self.models.expand) |m| m.deinit();

        // c.llama_backend_free();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Embedding
    // ═══════════════════════════════════════════════════════════════════════

    /// Generate embedding for text using embeddinggemma
    /// Returns 768-dimensional float vector
    pub fn embed(self: *LLM, text: []const u8) ![768]f32 {
        const model = try self.ensureModel(.embed);
        return model.embed(text);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Reranking
    // ═══════════════════════════════════════════════════════════════════════

    /// Rerank a document against a query
    /// Returns relevance score 0.0-1.0 based on yes/no logprobs
    pub fn rerank(self: *LLM, query: []const u8, document: []const u8) !f32 {
        const model = try self.ensureModel(.rerank);

        // Build reranker prompt
        const prompt = try std.fmt.allocPrint(
            self.allocator,
            \\<Instruct>: Given a search query, determine if the document is relevant.
            \\Respond only "yes" or "no".
            \\<Query>: {s}
            \\<Document>: {s}
        ,
            .{ query, document },
        );
        defer self.allocator.free(prompt);

        // Get logprobs for yes/no
        const result = try model.generateWithLogprobs(prompt, 1);

        // Convert to confidence score
        // P(yes) from logprobs
        return result.confidence;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Query Expansion
    // ═══════════════════════════════════════════════════════════════════════

    /// Generate query variations using fine-tuned expansion model
    pub fn expandQuery(self: *LLM, query: []const u8) ![][]const u8 {
        const model = try self.ensureModel(.expand);

        const prompt = try std.fmt.allocPrint(
            self.allocator,
            \\Generate 2 alternative search queries for: "{s}"
            \\Return only the queries, one per line.
        ,
            .{query},
        );
        defer self.allocator.free(prompt);

        const response = try model.generate(prompt, 150);

        // Parse response into individual queries
        var queries = std.ArrayList([]const u8).init(self.allocator);

        var lines = std.mem.splitSequence(u8, response, "\n");
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len > 0) {
                // Skip numbered prefixes like "1." or "2."
                var start: usize = 0;
                if (trimmed.len > 2 and trimmed[1] == '.' or trimmed[1] == ')') {
                    start = 2;
                }
                const cleaned = std.mem.trim(u8, trimmed[start..], " ");
                if (cleaned.len > 0) {
                    try queries.append(try self.allocator.dupe(u8, cleaned));
                }
            }
        }

        return queries.toOwnedSlice();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Model Management
    // ═══════════════════════════════════════════════════════════════════════

    const ModelType = enum { embed, rerank, expand };

    fn ensureModel(self: *LLM, model_type: ModelType) !*Model {
        const model_ptr = switch (model_type) {
            .embed => &self.models.embed,
            .rerank => &self.models.rerank,
            .expand => &self.models.expand,
        };

        if (model_ptr.*) |model| {
            return model;
        }

        // Load model
        const spec = switch (model_type) {
            .embed => DEFAULT_MODELS.EMBED,
            .rerank => DEFAULT_MODELS.RERANK,
            .expand => DEFAULT_MODELS.EXPAND,
        };

        const model_path = try self.ensureModelDownloaded(spec);
        const model = try Model.init(self.allocator, model_path, spec);
        model_ptr.* = model;

        return model;
    }

    fn ensureModelDownloaded(self: *LLM, spec: ModelSpec) ![]const u8 {
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

        return path;
    }

    fn downloadModel(self: *LLM, spec: ModelSpec, dest: []const u8) !void {
        const url = try std.fmt.allocPrint(
            self.allocator,
            "https://huggingface.co/{s}/resolve/main/{s}",
            .{ spec.hf_repo, spec.hf_file },
        );
        defer self.allocator.free(url);

        std.debug.print("Downloading model: {s}\n", .{spec.hf_file});
        std.debug.print("  From: {s}\n", .{url});
        std.debug.print("  To: {s}\n", .{dest});

        // Ensure directory exists
        if (std.fs.path.dirname(dest)) |dir| {
            std.fs.makeDirAbsolute(dir) catch |err| {
                if (err != error.PathAlreadyExists) return err;
            };
        }

        // Use curl for download (portable)
        const result = try std.process.Child.run(.{
            .allocator = self.allocator,
            .argv = &[_][]const u8{
                "curl",
                "-L",      // Follow redirects
                "-o",
                dest,
                "--progress-bar",
                url,
            },
        });

        if (result.term.Exited != 0) {
            std.debug.print("Download failed: {s}\n", .{result.stderr});
            return error.DownloadFailed;
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Model wrapper (llama.cpp abstraction)
// ═══════════════════════════════════════════════════════════════════════════

pub const Model = struct {
    allocator: std.mem.Allocator,
    path: []const u8,
    spec: LLM.ModelSpec,

    // llama.cpp handles (opaque when not linked)
    // model: *c.llama_model,
    // ctx: *c.llama_context,

    pub fn init(
        allocator: std.mem.Allocator,
        path: []const u8,
        spec: LLM.ModelSpec,
    ) !*Model {
        const self = try allocator.create(Model);
        self.* = .{
            .allocator = allocator,
            .path = path,
            .spec = spec,
        };

        // Load model via llama.cpp
        // var model_params = c.llama_model_default_params();
        // self.model = c.llama_load_model_from_file(path.ptr, model_params)
        //     orelse return error.ModelLoadFailed;
        //
        // var ctx_params = c.llama_context_default_params();
        // ctx_params.n_ctx = spec.n_ctx;
        // ctx_params.n_batch = spec.n_batch;
        // ctx_params.embedding = true;  // For embedding models
        //
        // self.ctx = c.llama_new_context_with_model(self.model, ctx_params)
        //     orelse return error.ContextCreateFailed;

        return self;
    }

    pub fn deinit(self: *Model) void {
        // c.llama_free(self.ctx);
        // c.llama_free_model(self.model);
        self.allocator.destroy(self);
    }

    /// Generate embeddings for text
    pub fn embed(self: *Model, text: []const u8) ![768]f32 {
        _ = self;
        _ = text;

        // Placeholder - actual implementation:
        //
        // // Tokenize
        // var tokens: [512]c.llama_token = undefined;
        // const n_tokens = c.llama_tokenize(
        //     self.model, text.ptr, @intCast(text.len),
        //     &tokens, tokens.len, true, false
        // );
        //
        // // Decode (forward pass)
        // const batch = c.llama_batch_get_one(&tokens, n_tokens, 0, 0);
        // if (c.llama_decode(self.ctx, batch) != 0) {
        //     return error.DecodeFailed;
        // }
        //
        // // Get embeddings from final layer
        // const embeddings = c.llama_get_embeddings(self.ctx);
        // var result: [768]f32 = undefined;
        // @memcpy(&result, embeddings[0..768]);
        //
        // return result;

        // Return dummy embedding for now
        var result: [768]f32 = undefined;
        @memset(&result, 0);
        return result;
    }

    /// Generate text completion
    pub fn generate(self: *Model, prompt: []const u8, max_tokens: u32) ![]const u8 {
        _ = self;
        _ = max_tokens;

        // Placeholder - actual implementation would use llama_decode loop
        return self.allocator.dupe(u8, prompt);
    }

    /// Generate with logprobs for reranking
    pub fn generateWithLogprobs(self: *Model, prompt: []const u8, max_tokens: u32) !LogprobResult {
        _ = self;
        _ = prompt;
        _ = max_tokens;

        // Placeholder - actual implementation:
        //
        // // Similar to generate, but extract logprobs
        // // For reranking, we only care about first token (yes/no)
        // // and its probability
        //
        // const logits = c.llama_get_logits(self.ctx);
        // const yes_token = c.llama_token_get_text(self.model, "yes");
        // const no_token = c.llama_token_get_text(self.model, "no");
        //
        // // Softmax to get probabilities
        // const yes_logit = logits[yes_token];
        // const no_logit = logits[no_token];
        // const yes_prob = std.math.exp(yes_logit) /
        //     (std.math.exp(yes_logit) + std.math.exp(no_logit));

        return .{
            .text = "yes",
            .confidence = 0.5,
        };
    }
};

pub const LogprobResult = struct {
    text: []const u8,
    confidence: f32,
};

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

test "LLM initialization" {
    // Just verify structs compile correctly
    _ = LLM.DEFAULT_MODELS.EMBED;
    _ = LLM.DEFAULT_MODELS.RERANK;
    _ = LLM.DEFAULT_MODELS.EXPAND;
}

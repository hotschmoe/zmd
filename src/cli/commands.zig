const std = @import("std");
const Database = @import("../storage/database.zig").Database;
const SearchPipeline = @import("../search/pipeline.zig").SearchPipeline;
const LLM = @import("../llm/engine.zig").LLM;

// ═══════════════════════════════════════════════════════════════════════════
// Output Formatting
// ═══════════════════════════════════════════════════════════════════════════

const Colors = struct {
    const reset = "\x1b[0m";
    const bold = "\x1b[1m";
    const dim = "\x1b[2m";
    const green = "\x1b[32m";
    const yellow = "\x1b[33m";
    const cyan = "\x1b[36m";
};

fn useColors() bool {
    return std.posix.getenv("NO_COLOR") == null;
}

fn colorize(text: []const u8, color: []const u8) []const u8 {
    if (!useColors()) return text;
    _ = color;
    // For now, return plain text - proper implementation would allocate
    return text;
}

// ═══════════════════════════════════════════════════════════════════════════
// Collection Commands
// ═══════════════════════════════════════════════════════════════════════════

pub fn handleCollection(
    allocator: std.mem.Allocator,
    db: *Database,
    args: []const []const u8,
) !void {
    if (args.len == 0) {
        std.debug.print("Usage: qmd collection <add|list|remove> [args]\n", .{});
        return;
    }

    const subcommand = args[0];

    if (std.mem.eql(u8, subcommand, "add")) {
        try handleCollectionAdd(allocator, db, args[1..]);
    } else if (std.mem.eql(u8, subcommand, "list")) {
        try handleCollectionList(allocator, db);
    } else if (std.mem.eql(u8, subcommand, "remove")) {
        try handleCollectionRemove(db, args[1..]);
    } else {
        std.debug.print("Unknown collection subcommand: {s}\n", .{subcommand});
    }
}

fn handleCollectionAdd(
    allocator: std.mem.Allocator,
    db: *Database,
    args: []const []const u8,
) !void {
    if (args.len == 0) {
        std.debug.print("Usage: qmd collection add <path> [--name <n>] [--mask <glob>]\n", .{});
        return;
    }

    const path = args[0];
    var name: []const u8 = std.fs.path.basename(path);
    var glob: []const u8 = "**/*.md";

    // Parse options
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--name") and i + 1 < args.len) {
            name = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--mask") and i + 1 < args.len) {
            glob = args[i + 1];
            i += 1;
        }
    }

    // Resolve absolute path
    const abs_path = try std.fs.realpathAlloc(allocator, path);
    defer allocator.free(abs_path);

    // Create collection
    const collection_id = try db.addCollection(name, abs_path, glob);
    std.debug.print("Created collection '{s}' (id: {d})\n", .{ name, collection_id });

    // Index files
    var indexed: usize = 0;
    var dir = try std.fs.openDirAbsolute(abs_path, .{ .iterate = true });
    defer dir.close();

    var walker = dir.walk(allocator) catch |err| {
        std.debug.print("Error walking directory: {}\n", .{err});
        return;
    };
    defer walker.deinit();

    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;

        // Check glob match (simplified - just .md extension for now)
        if (!std.mem.endsWith(u8, entry.basename, ".md")) continue;

        const file_path = try std.fs.path.join(allocator, &.{ abs_path, entry.path });
        defer allocator.free(file_path);

        // Read file content
        const content = std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024) catch |err| {
            std.debug.print("  Skipping {s}: {}\n", .{ entry.path, err });
            continue;
        };
        defer allocator.free(content);

        // Extract title (first heading)
        const title = extractTitle(content);

        // Compute content hash
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(content, &hash, .{});
        const hash_str = std.fmt.bytesToHex(hash[0..6], .lower);

        // Insert document
        _ = try db.insertDocument(
            collection_id,
            entry.path,
            title,
            content,
            &hash_str,
        );

        indexed += 1;
    }

    std.debug.print("Indexed {d} documents\n", .{indexed});
}

fn handleCollectionList(allocator: std.mem.Allocator, db: *Database) !void {
    const collections = try db.listCollections(allocator);

    if (collections.len == 0) {
        std.debug.print("No collections found.\n", .{});
        return;
    }

    const stdout = std.io.getStdOut().writer();

    try stdout.writeAll("Collections:\n");
    for (collections) |col| {
        try stdout.print("  {s}\n", .{col.name});
        try stdout.print("    Path: {s}\n", .{col.path});
        try stdout.print("    Glob: {s}\n\n", .{col.glob_mask});
    }
}

fn handleCollectionRemove(db: *Database, args: []const []const u8) !void {
    if (args.len == 0) {
        std.debug.print("Usage: qmd collection remove <name>\n", .{});
        return;
    }

    _ = db;
    const name = args[0];
    // TODO: Implement
    std.debug.print("Would remove collection: {s}\n", .{name});
}

// ═══════════════════════════════════════════════════════════════════════════
// Search Commands
// ═══════════════════════════════════════════════════════════════════════════

pub fn handleSearch(
    allocator: std.mem.Allocator,
    db: *Database,
    args: []const []const u8,
) !void {
    const opts = try parseSearchOptions(args);

    if (opts.query == null) {
        std.debug.print("Usage: qmd search <query> [options]\n", .{});
        return;
    }

    const results = try db.ftsSearch(allocator, opts.query.?, opts.limit);

    try outputResults(allocator, results, opts);
}

pub fn handleVectorSearch(
    allocator: std.mem.Allocator,
    db: *Database,
    args: []const []const u8,
) !void {
    _ = db;
    const opts = try parseSearchOptions(args);

    if (opts.query == null) {
        std.debug.print("Usage: qmd vsearch <query> [options]\n", .{});
        return;
    }

    // TODO: Initialize LLM and run vector search
    std.debug.print("Vector search for: {s}\n", .{opts.query.?});
    std.debug.print("Note: LLM not initialized - run with --init-llm\n", .{});

    _ = allocator;
}

pub fn handleHybridQuery(
    allocator: std.mem.Allocator,
    db: *Database,
    args: []const []const u8,
) !void {
    _ = db;
    const opts = try parseSearchOptions(args);

    if (opts.query == null) {
        std.debug.print("Usage: qmd query <query> [options]\n", .{});
        return;
    }

    // TODO: Full hybrid pipeline
    std.debug.print("Hybrid query for: {s}\n", .{opts.query.?});
    std.debug.print("Note: LLM not initialized - run with --init-llm\n", .{});

    _ = allocator;
}

const SearchOptions = struct {
    query: ?[]const u8 = null,
    limit: usize = 5,
    min_score: f32 = 0.0,
    full: bool = false,
    format: OutputFormat = .pretty,
};

const OutputFormat = enum { pretty, json, csv, files, md, xml };

fn parseSearchOptions(args: []const []const u8) !SearchOptions {
    var opts = SearchOptions{};

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "-n") and i + 1 < args.len) {
            opts.limit = try std.fmt.parseInt(usize, args[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--min-score") and i + 1 < args.len) {
            opts.min_score = try std.fmt.parseFloat(f32, args[i + 1]);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--full")) {
            opts.full = true;
        } else if (std.mem.eql(u8, arg, "--json")) {
            opts.format = .json;
        } else if (std.mem.eql(u8, arg, "--csv")) {
            opts.format = .csv;
        } else if (std.mem.eql(u8, arg, "--files")) {
            opts.format = .files;
        } else if (std.mem.eql(u8, arg, "--md")) {
            opts.format = .md;
        } else if (std.mem.eql(u8, arg, "--xml")) {
            opts.format = .xml;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            opts.query = arg;
        }
    }

    return opts;
}

fn outputResults(
    allocator: std.mem.Allocator,
    results: anytype,
    opts: SearchOptions,
) !void {
    const stdout = std.io.getStdOut().writer();

    switch (opts.format) {
        .pretty => {
            for (results) |r| {
                // Score color
                const score_pct = r.score * 100;
                if (useColors()) {
                    if (score_pct > 70) {
                        try stdout.print("{s}", .{Colors.green});
                    } else if (score_pct > 40) {
                        try stdout.print("{s}", .{Colors.yellow});
                    } else {
                        try stdout.print("{s}", .{Colors.dim});
                    }
                }

                try stdout.print("{s}\n", .{r.path});

                if (r.title) |title| {
                    try stdout.print("Title: {s}\n", .{title});
                }

                try stdout.print("Score: {d:.0}%\n", .{score_pct});

                if (r.snippet) |snippet| {
                    try stdout.print("\n{s}\n", .{snippet});
                }

                if (useColors()) {
                    try stdout.print("{s}", .{Colors.reset});
                }

                try stdout.writeByte('\n');
            }
        },

        .json => {
            try stdout.writeAll("[\n");
            for (results, 0..) |r, i| {
                try stdout.print(
                    \\  {{"path": "{s}", "title": {s}, "score": {d:.4}, "snippet": {s}}}
                ,
                    .{
                        r.path,
                        if (r.title) |t| try std.fmt.allocPrint(allocator, "\"{s}\"", .{t}) else "null",
                        r.score,
                        if (r.snippet) |s| try std.fmt.allocPrint(allocator, "\"{s}\"", .{s}) else "null",
                    },
                );
                if (i < results.len - 1) try stdout.writeByte(',');
                try stdout.writeByte('\n');
            }
            try stdout.writeAll("]\n");
        },

        .files => {
            for (results) |r| {
                try stdout.print("{d:.4},{s},{s}\n", .{
                    r.score,
                    r.path,
                    r.collection,
                });
            }
        },

        .csv => {
            try stdout.writeAll("score,path,title,collection\n");
            for (results) |r| {
                try stdout.print("{d:.4},{s},{s},{s}\n", .{
                    r.score,
                    r.path,
                    r.title orelse "",
                    r.collection,
                });
            }
        },

        .md => {
            for (results) |r| {
                try stdout.print("## {s}\n\n", .{r.title orelse r.path});
                try stdout.print("**Score:** {d:.0}%  \n", .{r.score * 100});
                try stdout.print("**Path:** `{s}`\n\n", .{r.path});
                if (r.snippet) |snippet| {
                    try stdout.print("{s}\n\n", .{snippet});
                }
                try stdout.writeAll("---\n\n");
            }
        },

        .xml => {
            try stdout.writeAll("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<results>\n");
            for (results) |r| {
                try stdout.print("  <result>\n", .{});
                try stdout.print("    <path>{s}</path>\n", .{r.path});
                if (r.title) |title| {
                    try stdout.print("    <title>{s}</title>\n", .{title});
                }
                try stdout.print("    <score>{d:.4}</score>\n", .{r.score});
                if (r.snippet) |snippet| {
                    try stdout.print("    <snippet>{s}</snippet>\n", .{snippet});
                }
                try stdout.print("  </result>\n", .{});
            }
            try stdout.writeAll("</results>\n");
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Other Commands
// ═══════════════════════════════════════════════════════════════════════════

pub fn handleEmbed(
    allocator: std.mem.Allocator,
    db: *Database,
    args: []const []const u8,
) !void {
    _ = allocator;
    _ = db;
    _ = args;

    std.debug.print("Embedding requires LLM initialization.\n", .{});
    std.debug.print("This will download models on first run:\n", .{});
    std.debug.print("  - embeddinggemma-300M (~1.6GB)\n", .{});
}

pub fn handleGet(
    allocator: std.mem.Allocator,
    db: *Database,
    args: []const []const u8,
) !void {
    _ = allocator;
    _ = db;

    if (args.len == 0) {
        std.debug.print("Usage: qmd get <file>\n", .{});
        return;
    }

    const path = args[0];
    // TODO: Implement document retrieval
    std.debug.print("Would retrieve: {s}\n", .{path});
}

pub fn handleStatus(allocator: std.mem.Allocator, db: *Database) !void {
    const doc_count = try db.getDocumentCount();
    const vec_count = try db.getVectorCount();
    const collections = try db.listCollections(allocator);

    const stdout = std.io.getStdOut().writer();

    try stdout.writeAll("QMD Index Status\n");
    try stdout.writeAll("═══════════════════════════════════════\n");
    try stdout.print("Documents:   {d}\n", .{doc_count});
    try stdout.print("Vectors:     {d}\n", .{vec_count});
    try stdout.print("Collections: {d}\n", .{collections.len});

    if (collections.len > 0) {
        try stdout.writeAll("\nCollections:\n");
        for (collections) |col| {
            try stdout.print("  • {s} ({s})\n", .{ col.name, col.path });
        }
    }
}

pub fn handleContext(
    allocator: std.mem.Allocator,
    db: *Database,
    args: []const []const u8,
) !void {
    _ = allocator;
    _ = db;

    if (args.len == 0) {
        std.debug.print("Usage: qmd context <add|list|rm> [args]\n", .{});
        return;
    }

    std.debug.print("Context command: {s}\n", .{args[0]});
    // TODO: Implement
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn extractTitle(content: []const u8) ?[]const u8 {
    var lines = std.mem.splitSequence(u8, content, "\n");

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");

        // ATX heading: # Title
        if (std.mem.startsWith(u8, trimmed, "#")) {
            var start: usize = 0;
            while (start < trimmed.len and trimmed[start] == '#') : (start += 1) {}
            const title = std.mem.trim(u8, trimmed[start..], " \t");
            if (title.len > 0) return title;
        }

        // Setext heading: check for underline
        if (lines.peek()) |next_line| {
            const next_trimmed = std.mem.trim(u8, next_line, " \t\r");
            if (next_trimmed.len > 0 and
                (std.mem.allEqual(u8, next_trimmed, '=') or
                std.mem.allEqual(u8, next_trimmed, '-')))
            {
                if (trimmed.len > 0) return trimmed;
            }
        }
    }

    return null;
}

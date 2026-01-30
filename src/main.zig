const std = @import("std");
const Database = @import("storage/database.zig").Database;
const SearchPipeline = @import("search/pipeline.zig").SearchPipeline;
const MCPServer = @import("mcp/server.zig").MCPServer;
const cli = @import("cli/commands.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsage();
        return;
    }

    const command = args[1];
    const cmd_args = args[2..];

    // Initialize database
    const cache_dir = try getCacheDir(allocator);
    defer allocator.free(cache_dir);

    const db_path = try std.fs.path.join(allocator, &.{ cache_dir, "index.sqlite" });
    defer allocator.free(db_path);

    if (std.mem.eql(u8, command, "mcp")) {
        // MCP server mode - no DB init needed until first command
        var server = try MCPServer.init(allocator, db_path);
        defer server.deinit();
        try server.run();
    } else {
        // CLI mode
        var db = try Database.init(allocator, db_path);
        defer db.deinit();

        if (std.mem.eql(u8, command, "collection")) {
            try cli.handleCollection(allocator, &db, cmd_args);
        } else if (std.mem.eql(u8, command, "search")) {
            try cli.handleSearch(allocator, &db, cmd_args);
        } else if (std.mem.eql(u8, command, "vsearch")) {
            try cli.handleVectorSearch(allocator, &db, cmd_args);
        } else if (std.mem.eql(u8, command, "query")) {
            try cli.handleHybridQuery(allocator, &db, cmd_args);
        } else if (std.mem.eql(u8, command, "embed")) {
            try cli.handleEmbed(allocator, &db, cmd_args);
        } else if (std.mem.eql(u8, command, "get")) {
            try cli.handleGet(allocator, &db, cmd_args);
        } else if (std.mem.eql(u8, command, "status")) {
            try cli.handleStatus(allocator, &db);
        } else if (std.mem.eql(u8, command, "context")) {
            try cli.handleContext(allocator, &db, cmd_args);
        } else {
            std.debug.print("Unknown command: {s}\n", .{command});
            try printUsage();
        }
    }
}

fn getCacheDir(allocator: std.mem.Allocator) ![]const u8 {
    // XDG_CACHE_HOME or ~/.cache/qmd
    if (std.posix.getenv("XDG_CACHE_HOME")) |xdg| {
        return std.fs.path.join(allocator, &.{ xdg, "qmd" });
    }

    if (std.posix.getenv("HOME")) |home| {
        return std.fs.path.join(allocator, &.{ home, ".cache", "qmd" });
    }

    return error.NoCacheDir;
}

fn printUsage() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll(
        \\qmd - Quick Markdown Search
        \\
        \\USAGE:
        \\    qmd <command> [options] [arguments]
        \\
        \\COMMANDS:
        \\    collection add <path>     Index markdown files from path
        \\    collection list           List all collections
        \\    collection remove <name>  Remove a collection
        \\
        \\    search <query>            BM25 full-text search
        \\    vsearch <query>           Vector semantic search
        \\    query <query>             Hybrid search with reranking
        \\
        \\    embed                     Generate vector embeddings
        \\    get <file>                Get document content
        \\    status                    Show index status
        \\
        \\    context add [path] "text" Add context description
        \\    context list              List all contexts
        \\
        \\    mcp                       Run MCP server (stdio)
        \\
        \\OPTIONS:
        \\    -n <num>          Number of results (default: 5)
        \\    --min-score <f>   Minimum score threshold
        \\    --full            Show full document content
        \\    --json            JSON output
        \\    --files           Output: score,filepath,context
        \\    --index <name>    Use named index
        \\
    );
}

test "basic functionality" {
    // Placeholder for tests
}

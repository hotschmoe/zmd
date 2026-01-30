const std = @import("std");
const json = std.json;
const Database = @import("../storage/database.zig").Database;
const SearchPipeline = @import("../search/pipeline.zig").SearchPipeline;
const LLM = @import("../llm/engine.zig").LLM;

/// MCP (Model Context Protocol) Server
/// Implements JSON-RPC 2.0 over stdio for Claude integration
pub const MCPServer = struct {
    allocator: std.mem.Allocator,
    db_path: []const u8,
    db: ?*Database = null,
    pipeline: ?*SearchPipeline = null,
    llm: ?*LLM = null,

    const SERVER_NAME = "qmd";
    const SERVER_VERSION = "0.1.0";
    const PROTOCOL_VERSION = "2024-11-05";

    pub fn init(allocator: std.mem.Allocator, db_path: []const u8) !MCPServer {
        return .{
            .allocator = allocator,
            .db_path = db_path,
        };
    }

    pub fn deinit(self: *MCPServer) void {
        if (self.pipeline) |p| {
            p.deinit();
            self.allocator.destroy(p);
        }
        if (self.db) |db| {
            db.deinit();
            self.allocator.destroy(db);
        }
        if (self.llm) |llm| {
            llm.deinit();
            self.allocator.destroy(llm);
        }
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

            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;

            const response = self.handleMessage(trimmed) catch |err| {
                const error_response = self.makeError(null, -32603, @errorName(err));
                try self.writeResponse(stdout, error_response);
                continue;
            };

            try self.writeResponse(stdout, response);
        }
    }

    fn handleMessage(self: *MCPServer, message: []const u8) !json.Value {
        var parsed = try json.parseFromSlice(json.Value, self.allocator, message, .{});
        defer parsed.deinit();

        const request = parsed.value;

        // Validate JSON-RPC structure
        const method = request.object.get("method") orelse
            return self.makeError(request.object.get("id"), -32600, "Invalid Request: missing method");

        const method_str = switch (method) {
            .string => |s| s,
            else => return self.makeError(request.object.get("id"), -32600, "Invalid Request: method must be string"),
        };

        const params = request.object.get("params");
        const id = request.object.get("id");

        // Route to handler
        if (std.mem.eql(u8, method_str, "initialize")) {
            return self.handleInitialize(params, id);
        } else if (std.mem.eql(u8, method_str, "tools/list")) {
            return self.handleToolsList(id);
        } else if (std.mem.eql(u8, method_str, "tools/call")) {
            return self.handleToolsCall(params, id);
        } else if (std.mem.eql(u8, method_str, "ping")) {
            return self.makeSuccess(id, .{ .object = std.json.ObjectMap.init(self.allocator) });
        } else {
            return self.makeError(id, -32601, "Method not found");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MCP Protocol Handlers
    // ═══════════════════════════════════════════════════════════════════════

    fn handleInitialize(self: *MCPServer, params: ?json.Value, id: ?json.Value) !json.Value {
        _ = params;

        // Initialize database on first use
        if (self.db == null) {
            const db = try self.allocator.create(Database);
            db.* = try Database.init(self.allocator, self.db_path);
            self.db = db;
        }

        var result = json.ObjectMap.init(self.allocator);

        // Protocol version
        try result.put("protocolVersion", .{ .string = PROTOCOL_VERSION });

        // Server info
        var server_info = json.ObjectMap.init(self.allocator);
        try server_info.put("name", .{ .string = SERVER_NAME });
        try server_info.put("version", .{ .string = SERVER_VERSION });
        try result.put("serverInfo", .{ .object = server_info });

        // Capabilities
        var capabilities = json.ObjectMap.init(self.allocator);
        var tools = json.ObjectMap.init(self.allocator);
        try capabilities.put("tools", .{ .object = tools });
        try result.put("capabilities", .{ .object = capabilities });

        return self.makeSuccess(id, .{ .object = result });
    }

    fn handleToolsList(self: *MCPServer, id: ?json.Value) !json.Value {
        var tools = json.Array.init(self.allocator);

        // qmd_search tool
        try tools.append(try self.makeToolDef(
            "qmd_search",
            "Fast BM25 keyword search across indexed markdown documents",
            &[_]ParamDef{
                .{ .name = "query", .type = "string", .description = "Search query", .required = true },
                .{ .name = "limit", .type = "integer", .description = "Max results (default: 5)", .required = false },
                .{ .name = "min_score", .type = "number", .description = "Minimum score threshold", .required = false },
            },
        ));

        // qmd_vsearch tool
        try tools.append(try self.makeToolDef(
            "qmd_vsearch",
            "Semantic vector search for conceptually similar documents",
            &[_]ParamDef{
                .{ .name = "query", .type = "string", .description = "Search query", .required = true },
                .{ .name = "limit", .type = "integer", .description = "Max results (default: 5)", .required = false },
                .{ .name = "min_score", .type = "number", .description = "Minimum score threshold", .required = false },
            },
        ));

        // qmd_query tool
        try tools.append(try self.makeToolDef(
            "qmd_query",
            "Hybrid search with query expansion and LLM reranking (best quality)",
            &[_]ParamDef{
                .{ .name = "query", .type = "string", .description = "Search query", .required = true },
                .{ .name = "limit", .type = "integer", .description = "Max results (default: 5)", .required = false },
                .{ .name = "min_score", .type = "number", .description = "Minimum score threshold", .required = false },
            },
        ));

        // qmd_get tool
        try tools.append(try self.makeToolDef(
            "qmd_get",
            "Retrieve document content by path or docid",
            &[_]ParamDef{
                .{ .name = "path", .type = "string", .description = "Document path or docid (#abc123)", .required = true },
                .{ .name = "full", .type = "boolean", .description = "Return full content", .required = false },
            },
        ));

        // qmd_status tool
        try tools.append(try self.makeToolDef(
            "qmd_status",
            "Show index health and collection information",
            &[_]ParamDef{},
        ));

        var result = json.ObjectMap.init(self.allocator);
        try result.put("tools", .{ .array = tools });

        return self.makeSuccess(id, .{ .object = result });
    }

    fn handleToolsCall(self: *MCPServer, params: ?json.Value, id: ?json.Value) !json.Value {
        const p = params orelse return self.makeError(id, -32602, "Invalid params");

        const name = p.object.get("name") orelse
            return self.makeError(id, -32602, "Missing tool name");
        const name_str = switch (name) {
            .string => |s| s,
            else => return self.makeError(id, -32602, "Tool name must be string"),
        };

        const arguments = p.object.get("arguments");

        // Ensure database is initialized
        const db = self.db orelse {
            const new_db = try self.allocator.create(Database);
            new_db.* = try Database.init(self.allocator, self.db_path);
            self.db = new_db;
            return self.handleToolsCall(params, id);
        };

        // Route to tool handler
        if (std.mem.eql(u8, name_str, "qmd_search")) {
            return self.toolSearch(db, arguments, id);
        } else if (std.mem.eql(u8, name_str, "qmd_vsearch")) {
            return self.toolVectorSearch(db, arguments, id);
        } else if (std.mem.eql(u8, name_str, "qmd_query")) {
            return self.toolHybridQuery(db, arguments, id);
        } else if (std.mem.eql(u8, name_str, "qmd_status")) {
            return self.toolStatus(db, id);
        } else if (std.mem.eql(u8, name_str, "qmd_get")) {
            return self.toolGet(db, arguments, id);
        } else {
            return self.makeError(id, -32601, "Unknown tool");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Tool Implementations
    // ═══════════════════════════════════════════════════════════════════════

    fn toolSearch(self: *MCPServer, db: *Database, args: ?json.Value, id: ?json.Value) !json.Value {
        const query = self.getStringArg(args, "query") orelse
            return self.makeError(id, -32602, "Missing query parameter");

        const limit = self.getIntArg(args, "limit") orelse 5;

        const results = try db.ftsSearch(self.allocator, query, @intCast(limit));

        return self.makeToolResult(id, results);
    }

    fn toolVectorSearch(self: *MCPServer, db: *Database, args: ?json.Value, id: ?json.Value) !json.Value {
        _ = db;
        const query = self.getStringArg(args, "query") orelse
            return self.makeError(id, -32602, "Missing query parameter");
        _ = query;

        // TODO: Implement with LLM for embeddings
        return self.makeError(id, -32603, "Vector search requires LLM initialization");
    }

    fn toolHybridQuery(self: *MCPServer, db: *Database, args: ?json.Value, id: ?json.Value) !json.Value {
        _ = db;
        const query = self.getStringArg(args, "query") orelse
            return self.makeError(id, -32602, "Missing query parameter");
        _ = query;

        // TODO: Implement with full pipeline
        return self.makeError(id, -32603, "Hybrid query requires LLM initialization");
    }

    fn toolStatus(self: *MCPServer, db: *Database, id: ?json.Value) !json.Value {
        const doc_count = try db.getDocumentCount();
        const vec_count = try db.getVectorCount();
        const collections = try db.listCollections(self.allocator);

        var result = json.ObjectMap.init(self.allocator);
        try result.put("documents", .{ .integer = doc_count });
        try result.put("vectors", .{ .integer = vec_count });
        try result.put("collections", .{ .integer = @intCast(collections.len) });

        var content = json.Array.init(self.allocator);
        var text_obj = json.ObjectMap.init(self.allocator);
        try text_obj.put("type", .{ .string = "text" });
        try text_obj.put("text", .{ .string = try std.fmt.allocPrint(
            self.allocator,
            "QMD Index Status:\n  Documents: {d}\n  Vectors: {d}\n  Collections: {d}",
            .{ doc_count, vec_count, collections.len },
        ) });
        try content.append(.{ .object = text_obj });

        var wrapper = json.ObjectMap.init(self.allocator);
        try wrapper.put("content", .{ .array = content });

        return self.makeSuccess(id, .{ .object = wrapper });
    }

    fn toolGet(self: *MCPServer, db: *Database, args: ?json.Value, id: ?json.Value) !json.Value {
        _ = db;
        _ = args;
        // TODO: Implement document retrieval
        return self.makeError(id, -32603, "Not implemented");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════

    fn makeToolResult(self: *MCPServer, id: ?json.Value, results: anytype) !json.Value {
        var content = json.Array.init(self.allocator);

        // Format results as text
        var text = std.ArrayList(u8).init(self.allocator);
        const writer = text.writer();

        for (results, 0..) |r, i| {
            try writer.print("{d}. {s}", .{ i + 1, r.path });
            if (r.title) |title| {
                try writer.print(" - {s}", .{title});
            }
            try writer.print("\n   Score: {d:.2}\n", .{r.score});
            if (r.snippet) |snippet| {
                try writer.print("   {s}\n", .{snippet});
            }
            try writer.writeByte('\n');
        }

        var text_obj = json.ObjectMap.init(self.allocator);
        try text_obj.put("type", .{ .string = "text" });
        try text_obj.put("text", .{ .string = text.items });
        try content.append(.{ .object = text_obj });

        var wrapper = json.ObjectMap.init(self.allocator);
        try wrapper.put("content", .{ .array = content });

        return self.makeSuccess(id, .{ .object = wrapper });
    }

    const ParamDef = struct {
        name: []const u8,
        type: []const u8,
        description: []const u8,
        required: bool,
    };

    fn makeToolDef(self: *MCPServer, name: []const u8, description: []const u8, params: []const ParamDef) !json.Value {
        var tool = json.ObjectMap.init(self.allocator);
        try tool.put("name", .{ .string = name });
        try tool.put("description", .{ .string = description });

        // Input schema
        var input_schema = json.ObjectMap.init(self.allocator);
        try input_schema.put("type", .{ .string = "object" });

        var properties = json.ObjectMap.init(self.allocator);
        var required = json.Array.init(self.allocator);

        for (params) |param| {
            var prop = json.ObjectMap.init(self.allocator);
            try prop.put("type", .{ .string = param.type });
            try prop.put("description", .{ .string = param.description });
            try properties.put(param.name, .{ .object = prop });

            if (param.required) {
                try required.append(.{ .string = param.name });
            }
        }

        try input_schema.put("properties", .{ .object = properties });
        try input_schema.put("required", .{ .array = required });

        try tool.put("inputSchema", .{ .object = input_schema });

        return .{ .object = tool };
    }

    fn makeSuccess(self: *MCPServer, id: ?json.Value, result: json.Value) json.Value {
        var obj = json.ObjectMap.init(self.allocator);
        obj.put("jsonrpc", .{ .string = "2.0" }) catch unreachable;
        obj.put("id", id orelse .null) catch unreachable;
        obj.put("result", result) catch unreachable;
        return .{ .object = obj };
    }

    fn makeError(self: *MCPServer, id: ?json.Value, code: i32, message: []const u8) json.Value {
        var err = json.ObjectMap.init(self.allocator);
        err.put("code", .{ .integer = code }) catch unreachable;
        err.put("message", .{ .string = message }) catch unreachable;

        var obj = json.ObjectMap.init(self.allocator);
        obj.put("jsonrpc", .{ .string = "2.0" }) catch unreachable;
        obj.put("id", id orelse .null) catch unreachable;
        obj.put("error", .{ .object = err }) catch unreachable;

        return .{ .object = obj };
    }

    fn writeResponse(self: *MCPServer, writer: anytype, response: json.Value) !void {
        _ = self;
        try json.stringify(response, .{}, writer);
        try writer.writeByte('\n');
    }

    fn getStringArg(self: *MCPServer, args: ?json.Value, name: []const u8) ?[]const u8 {
        _ = self;
        const a = args orelse return null;
        const val = a.object.get(name) orelse return null;
        return switch (val) {
            .string => |s| s,
            else => null,
        };
    }

    fn getIntArg(self: *MCPServer, args: ?json.Value, name: []const u8) ?i64 {
        _ = self;
        const a = args orelse return null;
        const val = a.object.get(name) orelse return null;
        return switch (val) {
            .integer => |i| i,
            else => null,
        };
    }
};

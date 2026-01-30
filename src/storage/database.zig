const std = @import("std");
const c = @cImport({
    @cInclude("sqlite3.h");
});

pub const Database = struct {
    db: *c.sqlite3,
    allocator: std.mem.Allocator,

    // Prepared statements cache
    stmts: struct {
        insert_doc: ?*c.sqlite3_stmt = null,
        search_fts: ?*c.sqlite3_stmt = null,
        get_doc: ?*c.sqlite3_stmt = null,
        insert_vector: ?*c.sqlite3_stmt = null,
    } = .{},

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !Database {
        var db: ?*c.sqlite3 = null;

        const path_z = try allocator.dupeZ(u8, path);
        defer allocator.free(path_z);

        // Ensure directory exists
        if (std.fs.path.dirname(path)) |dir| {
            std.fs.makeDirAbsolute(dir) catch |err| {
                if (err != error.PathAlreadyExists) return err;
            };
        }

        const rc = c.sqlite3_open(path_z.ptr, &db);
        if (rc != c.SQLITE_OK) {
            if (db) |d| c.sqlite3_close(d);
            return error.DatabaseOpenFailed;
        }

        var self = Database{
            .db = db.?,
            .allocator = allocator,
        };

        try self.initSchema();
        return self;
    }

    pub fn deinit(self: *Database) void {
        // Finalize prepared statements
        inline for (std.meta.fields(@TypeOf(self.stmts))) |field| {
            if (@field(self.stmts, field.name)) |stmt| {
                _ = c.sqlite3_finalize(stmt);
            }
        }

        _ = c.sqlite3_close(self.db);
    }

    fn initSchema(self: *Database) !void {
        const schema =
            \\-- Collections (indexed directories)
            \\CREATE TABLE IF NOT EXISTS collections (
            \\    id INTEGER PRIMARY KEY,
            \\    name TEXT UNIQUE NOT NULL,
            \\    path TEXT NOT NULL,
            \\    glob_mask TEXT DEFAULT '**/*.md',
            \\    created_at TEXT DEFAULT (datetime('now')),
            \\    updated_at TEXT DEFAULT (datetime('now'))
            \\);
            \\
            \\-- Path contexts
            \\CREATE TABLE IF NOT EXISTS path_contexts (
            \\    id INTEGER PRIMARY KEY,
            \\    path TEXT UNIQUE NOT NULL,
            \\    context TEXT NOT NULL
            \\);
            \\
            \\-- Documents
            \\CREATE TABLE IF NOT EXISTS documents (
            \\    id INTEGER PRIMARY KEY,
            \\    collection_id INTEGER NOT NULL REFERENCES collections(id),
            \\    hash TEXT NOT NULL,
            \\    path TEXT NOT NULL,
            \\    title TEXT,
            \\    content TEXT NOT NULL,
            \\    created_at TEXT DEFAULT (datetime('now')),
            \\    updated_at TEXT DEFAULT (datetime('now')),
            \\    UNIQUE(collection_id, path)
            \\);
            \\
            \\-- FTS5 virtual table for full-text search
            \\CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            \\    title,
            \\    content,
            \\    content=documents,
            \\    content_rowid=id,
            \\    tokenize='porter unicode61'
            \\);
            \\
            \\-- Triggers to keep FTS in sync
            \\CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            \\    INSERT INTO documents_fts(rowid, title, content)
            \\    VALUES (new.id, new.title, new.content);
            \\END;
            \\
            \\CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
            \\    INSERT INTO documents_fts(documents_fts, rowid, title, content)
            \\    VALUES ('delete', old.id, old.title, old.content);
            \\END;
            \\
            \\CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
            \\    INSERT INTO documents_fts(documents_fts, rowid, title, content)
            \\    VALUES ('delete', old.id, old.title, old.content);
            \\    INSERT INTO documents_fts(rowid, title, content)
            \\    VALUES (new.id, new.title, new.content);
            \\END;
            \\
            \\-- Vector embeddings (chunked)
            \\CREATE TABLE IF NOT EXISTS content_vectors (
            \\    id INTEGER PRIMARY KEY,
            \\    doc_id INTEGER NOT NULL REFERENCES documents(id),
            \\    hash TEXT NOT NULL,
            \\    seq INTEGER NOT NULL,
            \\    start_pos INTEGER NOT NULL,
            \\    end_pos INTEGER NOT NULL,
            \\    embedding BLOB NOT NULL,
            \\    UNIQUE(doc_id, seq)
            \\);
            \\
            \\CREATE INDEX IF NOT EXISTS idx_vectors_doc ON content_vectors(doc_id);
            \\CREATE INDEX IF NOT EXISTS idx_vectors_hash ON content_vectors(hash);
        ;

        try self.execMultiple(schema);
    }

    fn execMultiple(self: *Database, sql: []const u8) !void {
        var err_msg: [*c]u8 = null;
        const rc = c.sqlite3_exec(self.db, sql.ptr, null, null, &err_msg);
        if (rc != c.SQLITE_OK) {
            if (err_msg) |msg| {
                std.debug.print("SQL error: {s}\n", .{msg});
                c.sqlite3_free(msg);
            }
            return error.SqlError;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Collection Operations
    // ═══════════════════════════════════════════════════════════════════════

    pub fn addCollection(self: *Database, name: []const u8, path: []const u8, glob: []const u8) !i64 {
        const sql = "INSERT INTO collections (name, path, glob_mask) VALUES (?, ?, ?) RETURNING id";
        var stmt: ?*c.sqlite3_stmt = null;

        if (c.sqlite3_prepare_v2(self.db, sql, -1, &stmt, null) != c.SQLITE_OK) {
            return error.PrepareError;
        }
        defer _ = c.sqlite3_finalize(stmt);

        _ = c.sqlite3_bind_text(stmt, 1, name.ptr, @intCast(name.len), c.SQLITE_TRANSIENT);
        _ = c.sqlite3_bind_text(stmt, 2, path.ptr, @intCast(path.len), c.SQLITE_TRANSIENT);
        _ = c.sqlite3_bind_text(stmt, 3, glob.ptr, @intCast(glob.len), c.SQLITE_TRANSIENT);

        if (c.sqlite3_step(stmt) != c.SQLITE_ROW) {
            return error.InsertFailed;
        }

        return c.sqlite3_column_int64(stmt, 0);
    }

    pub fn listCollections(self: *Database, allocator: std.mem.Allocator) ![]Collection {
        const sql = "SELECT id, name, path, glob_mask FROM collections ORDER BY name";
        var stmt: ?*c.sqlite3_stmt = null;

        if (c.sqlite3_prepare_v2(self.db, sql, -1, &stmt, null) != c.SQLITE_OK) {
            return error.PrepareError;
        }
        defer _ = c.sqlite3_finalize(stmt);

        var list = std.ArrayList(Collection).init(allocator);

        while (c.sqlite3_step(stmt) == c.SQLITE_ROW) {
            try list.append(.{
                .id = c.sqlite3_column_int64(stmt, 0),
                .name = try self.columnText(allocator, stmt, 1),
                .path = try self.columnText(allocator, stmt, 2),
                .glob_mask = try self.columnText(allocator, stmt, 3),
            });
        }

        return list.toOwnedSlice();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Document Operations
    // ═══════════════════════════════════════════════════════════════════════

    pub fn insertDocument(
        self: *Database,
        collection_id: i64,
        path: []const u8,
        title: ?[]const u8,
        content: []const u8,
        hash: []const u8,
    ) !i64 {
        const sql =
            \\INSERT INTO documents (collection_id, path, title, content, hash)
            \\VALUES (?, ?, ?, ?, ?)
            \\ON CONFLICT(collection_id, path) DO UPDATE SET
            \\    title = excluded.title,
            \\    content = excluded.content,
            \\    hash = excluded.hash,
            \\    updated_at = datetime('now')
            \\RETURNING id
        ;

        var stmt: ?*c.sqlite3_stmt = null;
        if (c.sqlite3_prepare_v2(self.db, sql, -1, &stmt, null) != c.SQLITE_OK) {
            return error.PrepareError;
        }
        defer _ = c.sqlite3_finalize(stmt);

        _ = c.sqlite3_bind_int64(stmt, 1, collection_id);
        _ = c.sqlite3_bind_text(stmt, 2, path.ptr, @intCast(path.len), c.SQLITE_TRANSIENT);
        if (title) |t| {
            _ = c.sqlite3_bind_text(stmt, 3, t.ptr, @intCast(t.len), c.SQLITE_TRANSIENT);
        } else {
            _ = c.sqlite3_bind_null(stmt, 3);
        }
        _ = c.sqlite3_bind_text(stmt, 4, content.ptr, @intCast(content.len), c.SQLITE_TRANSIENT);
        _ = c.sqlite3_bind_text(stmt, 5, hash.ptr, @intCast(hash.len), c.SQLITE_TRANSIENT);

        if (c.sqlite3_step(stmt) != c.SQLITE_ROW) {
            return error.InsertFailed;
        }

        return c.sqlite3_column_int64(stmt, 0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Search Operations
    // ═══════════════════════════════════════════════════════════════════════

    pub fn ftsSearch(
        self: *Database,
        allocator: std.mem.Allocator,
        query: []const u8,
        limit: usize,
    ) ![]SearchResult {
        // BM25 search with FTS5
        // Note: FTS5 bm25() returns negative scores (more negative = better)
        const sql =
            \\SELECT
            \\    d.id,
            \\    d.path,
            \\    d.title,
            \\    d.hash,
            \\    c.name as collection,
            \\    ABS(bm25(documents_fts)) as score,
            \\    snippet(documents_fts, 1, '**', '**', '...', 32) as snippet
            \\FROM documents_fts
            \\JOIN documents d ON documents_fts.rowid = d.id
            \\JOIN collections c ON d.collection_id = c.id
            \\WHERE documents_fts MATCH ?
            \\ORDER BY bm25(documents_fts)
            \\LIMIT ?
        ;

        var stmt: ?*c.sqlite3_stmt = null;
        if (c.sqlite3_prepare_v2(self.db, sql, -1, &stmt, null) != c.SQLITE_OK) {
            const err = c.sqlite3_errmsg(self.db);
            std.debug.print("FTS prepare error: {s}\n", .{err});
            return error.PrepareError;
        }
        defer _ = c.sqlite3_finalize(stmt);

        // Escape query for FTS5
        const escaped_query = try self.escapeFtsQuery(allocator, query);
        defer allocator.free(escaped_query);

        _ = c.sqlite3_bind_text(stmt, 1, escaped_query.ptr, @intCast(escaped_query.len), c.SQLITE_TRANSIENT);
        _ = c.sqlite3_bind_int(stmt, 2, @intCast(limit));

        var results = std.ArrayList(SearchResult).init(allocator);

        while (c.sqlite3_step(stmt) == c.SQLITE_ROW) {
            try results.append(.{
                .doc_id = c.sqlite3_column_int64(stmt, 0),
                .path = try self.columnText(allocator, stmt, 1),
                .title = try self.columnTextOptional(allocator, stmt, 2),
                .hash = try self.columnText(allocator, stmt, 3),
                .collection = try self.columnText(allocator, stmt, 4),
                .score = @floatCast(c.sqlite3_column_double(stmt, 5)),
                .snippet = try self.columnTextOptional(allocator, stmt, 6),
            });
        }

        return results.toOwnedSlice();
    }

    fn escapeFtsQuery(self: *Database, allocator: std.mem.Allocator, query: []const u8) ![]const u8 {
        _ = self;
        // Simple escaping: wrap terms in quotes if they contain special chars
        // Full implementation would handle operators like AND, OR, NOT, etc.
        var result = std.ArrayList(u8).init(allocator);

        var terms = std.mem.tokenizeAny(u8, query, " \t\n");
        var first = true;
        while (terms.next()) |term| {
            if (!first) try result.append(' ');
            first = false;

            // Check for FTS5 operators we want to preserve
            if (std.mem.eql(u8, term, "AND") or
                std.mem.eql(u8, term, "OR") or
                std.mem.eql(u8, term, "NOT"))
            {
                try result.appendSlice(term);
            } else {
                // Add wildcard for prefix matching
                try result.appendSlice(term);
                try result.append('*');
            }
        }

        return result.toOwnedSlice();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Vector Operations
    // ═══════════════════════════════════════════════════════════════════════

    pub fn insertVector(
        self: *Database,
        doc_id: i64,
        hash: []const u8,
        seq: u16,
        start_pos: u32,
        end_pos: u32,
        embedding: []const f32,
    ) !void {
        const sql =
            \\INSERT INTO content_vectors (doc_id, hash, seq, start_pos, end_pos, embedding)
            \\VALUES (?, ?, ?, ?, ?, ?)
            \\ON CONFLICT(doc_id, seq) DO UPDATE SET
            \\    embedding = excluded.embedding
        ;

        var stmt: ?*c.sqlite3_stmt = null;
        if (c.sqlite3_prepare_v2(self.db, sql, -1, &stmt, null) != c.SQLITE_OK) {
            return error.PrepareError;
        }
        defer _ = c.sqlite3_finalize(stmt);

        _ = c.sqlite3_bind_int64(stmt, 1, doc_id);
        _ = c.sqlite3_bind_text(stmt, 2, hash.ptr, @intCast(hash.len), c.SQLITE_TRANSIENT);
        _ = c.sqlite3_bind_int(stmt, 3, seq);
        _ = c.sqlite3_bind_int(stmt, 4, @intCast(start_pos));
        _ = c.sqlite3_bind_int(stmt, 5, @intCast(end_pos));
        _ = c.sqlite3_bind_blob(
            stmt,
            6,
            @ptrCast(embedding.ptr),
            @intCast(embedding.len * @sizeOf(f32)),
            c.SQLITE_TRANSIENT,
        );

        if (c.sqlite3_step(stmt) != c.SQLITE_DONE) {
            return error.InsertFailed;
        }
    }

    pub fn getAllVectors(self: *Database, allocator: std.mem.Allocator) ![]VectorRecord {
        const sql =
            \\SELECT cv.id, cv.doc_id, cv.seq, cv.embedding, d.path, d.title
            \\FROM content_vectors cv
            \\JOIN documents d ON cv.doc_id = d.id
        ;

        var stmt: ?*c.sqlite3_stmt = null;
        if (c.sqlite3_prepare_v2(self.db, sql, -1, &stmt, null) != c.SQLITE_OK) {
            return error.PrepareError;
        }
        defer _ = c.sqlite3_finalize(stmt);

        var results = std.ArrayList(VectorRecord).init(allocator);

        while (c.sqlite3_step(stmt) == c.SQLITE_ROW) {
            const blob = c.sqlite3_column_blob(stmt, 3);
            const blob_size = c.sqlite3_column_bytes(stmt, 3);
            const float_count = @divExact(@as(usize, @intCast(blob_size)), @sizeOf(f32));

            var embedding = try allocator.alloc(f32, float_count);
            @memcpy(embedding, @as([*]const f32, @ptrCast(@alignCast(blob)))[0..float_count]);

            try results.append(.{
                .id = c.sqlite3_column_int64(stmt, 0),
                .doc_id = c.sqlite3_column_int64(stmt, 1),
                .seq = @intCast(c.sqlite3_column_int(stmt, 2)),
                .embedding = embedding,
                .path = try self.columnText(allocator, stmt, 4),
                .title = try self.columnTextOptional(allocator, stmt, 5),
            });
        }

        return results.toOwnedSlice();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Utilities
    // ═══════════════════════════════════════════════════════════════════════

    fn columnText(self: *Database, allocator: std.mem.Allocator, stmt: *c.sqlite3_stmt, col: c_int) ![]const u8 {
        _ = self;
        const text = c.sqlite3_column_text(stmt, col);
        if (text == null) return "";
        const len = c.sqlite3_column_bytes(stmt, col);
        return try allocator.dupe(u8, text[0..@intCast(len)]);
    }

    fn columnTextOptional(self: *Database, allocator: std.mem.Allocator, stmt: *c.sqlite3_stmt, col: c_int) !?[]const u8 {
        _ = self;
        const text = c.sqlite3_column_text(stmt, col);
        if (text == null) return null;
        const len = c.sqlite3_column_bytes(stmt, col);
        return try allocator.dupe(u8, text[0..@intCast(len)]);
    }

    pub fn getDocumentCount(self: *Database) !i64 {
        const sql = "SELECT COUNT(*) FROM documents";
        var stmt: ?*c.sqlite3_stmt = null;

        if (c.sqlite3_prepare_v2(self.db, sql, -1, &stmt, null) != c.SQLITE_OK) {
            return error.PrepareError;
        }
        defer _ = c.sqlite3_finalize(stmt);

        if (c.sqlite3_step(stmt) != c.SQLITE_ROW) {
            return error.QueryFailed;
        }

        return c.sqlite3_column_int64(stmt, 0);
    }

    pub fn getVectorCount(self: *Database) !i64 {
        const sql = "SELECT COUNT(*) FROM content_vectors";
        var stmt: ?*c.sqlite3_stmt = null;

        if (c.sqlite3_prepare_v2(self.db, sql, -1, &stmt, null) != c.SQLITE_OK) {
            return error.PrepareError;
        }
        defer _ = c.sqlite3_finalize(stmt);

        if (c.sqlite3_step(stmt) != c.SQLITE_ROW) {
            return error.QueryFailed;
        }

        return c.sqlite3_column_int64(stmt, 0);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════

pub const Collection = struct {
    id: i64,
    name: []const u8,
    path: []const u8,
    glob_mask: []const u8,
};

pub const SearchResult = struct {
    doc_id: i64,
    path: []const u8,
    title: ?[]const u8,
    hash: []const u8,
    collection: []const u8,
    score: f32,
    snippet: ?[]const u8,
};

pub const VectorRecord = struct {
    id: i64,
    doc_id: i64,
    seq: u16,
    embedding: []f32,
    path: []const u8,
    title: ?[]const u8,
};

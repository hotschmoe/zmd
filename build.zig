const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ═══════════════════════════════════════════════════════════════════════
    // Main executable
    // ═══════════════════════════════════════════════════════════════════════
    const exe = b.addExecutable(.{
        .name = "qmd",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // SQLite with FTS5
    // ═══════════════════════════════════════════════════════════════════════
    // Option 1: Use vendored amalgamation (recommended for FTS5 control)
    exe.addCSourceFile(.{
        .file = b.path("vendor/sqlite3.c"),
        .flags = &.{
            "-DSQLITE_ENABLE_FTS5",
            "-DSQLITE_ENABLE_JSON1",
            "-DSQLITE_ENABLE_RTREE",
            "-DSQLITE_DQS=0",
            "-DSQLITE_DEFAULT_MEMSTATUS=0",
            "-DSQLITE_DEFAULT_WAL_SYNCHRONOUS=1",
            "-DSQLITE_LIKE_DOESNT_MATCH_BLOBS",
            "-DSQLITE_MAX_EXPR_DEPTH=0",
            "-DSQLITE_OMIT_DEPRECATED",
            "-DSQLITE_OMIT_PROGRESS_CALLBACK",
            "-DSQLITE_OMIT_SHARED_CACHE",
            "-DSQLITE_USE_ALLOCA",
            "-DSQLITE_THREADSAFE=0", // Single-threaded for CLI
        },
    });
    exe.addIncludePath(b.path("vendor"));

    // ═══════════════════════════════════════════════════════════════════════
    // llama.cpp integration
    // ═══════════════════════════════════════════════════════════════════════
    // Option A: System-installed llama.cpp
    // exe.linkSystemLibrary("llama");

    // Option B: Build from source (requires llama.cpp as submodule/dependency)
    // This would be the full integration - for now we stub it
    // const llama = b.dependency("llama_cpp", .{ .target = target, .optimize = optimize });
    // exe.linkLibrary(llama.artifact("llama"));

    // ═══════════════════════════════════════════════════════════════════════
    // Platform-specific linking
    // ═══════════════════════════════════════════════════════════════════════
    exe.linkLibC();

    switch (target.result.os.tag) {
        .macos => {
            // Metal acceleration for llama.cpp
            exe.linkFramework("Accelerate");
            exe.linkFramework("Metal");
            exe.linkFramework("MetalKit");
            exe.linkFramework("Foundation");
        },
        .linux => {
            // OpenBLAS for SIMD acceleration
            // exe.linkSystemLibrary("openblas");
        },
        else => {},
    }

    b.installArtifact(exe);

    // ═══════════════════════════════════════════════════════════════════════
    // Run step
    // ═══════════════════════════════════════════════════════════════════════
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run qmd");
    run_step.dependOn(&run_cmd.step);

    // ═══════════════════════════════════════════════════════════════════════
    // Tests
    // ═══════════════════════════════════════════════════════════════════════
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}

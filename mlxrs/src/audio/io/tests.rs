use super::*;

/// Bounded compressed decode: `reserve_under_cap` is the single
/// gate every decoded buffer passes through before its samples are
/// appended. A buffer that would push `out` past `cap` — the exact
/// scenario a compressed header under-estimating the cap creates (header
/// reserves `cap`, the decoder yields one more valid sample) — must
/// return a recoverable `Error::BoundedDecode`, and must NOT have grown `out`
/// past `cap` (no infallible `Vec` regrowth, no allocator abort).
#[test]
fn reserve_under_cap_rejects_buffer_that_would_exceed_cap() {
  // Small synthetic cap so the test allocates nothing large. `out` is
  // pre-filled to exactly one below the cap, mirroring "header reserved
  // up to `cap`, decoder produced one more sample than fits".
  let cap = 8usize;
  let mut out: Vec<f32> = vec![0.0; cap - 1];
  let cap_before = out.capacity();

  // A single extra sample exactly fills the cap — allowed.
  assert!(reserve_under_cap(&mut out, 1, cap).is_ok());
  // (We do not push here; we only assert the reservation/cap math.)

  // From the now-full-capacity position, any further buffer (even one
  // sample) must be rejected, NOT pushed through an infallible regrowth.
  out.resize(cap, 0.0); // out.len() == cap now (the "decoded up to cap" state)
  let r = reserve_under_cap(&mut out, 1, cap);
  assert!(
    matches!(r, Err(Error::BoundedDecode(_))),
    "over-cap buffer must return a recoverable BoundedDecode error, got {r:?}"
  );
  // The Vec must not have been grown past the cap by the rejected call.
  assert!(
    out.len() <= cap,
    "out grew past cap on the rejected path: len={} cap={cap}",
    out.len()
  );
  // Capacity sanity: the rejection happened before any reallocation that
  // would push capacity wildly above the cap (it may be >= cap from the
  // earlier successful reserve, but the reject path added nothing).
  assert!(
    out.capacity() >= cap_before,
    "capacity unexpectedly shrank: {} < {cap_before}",
    out.capacity()
  );
}

/// A buffer larger than the entire remaining room (multi-sample
/// over-cap, the realistic decoded-packet case) is rejected up front
/// with no growth.
#[test]
fn reserve_under_cap_rejects_oversized_buffer_against_empty_out() {
  let cap = 16usize;
  let mut out: Vec<f32> = Vec::new();
  // A packet claiming more samples than the whole cap.
  let r = reserve_under_cap(&mut out, cap + 1, cap);
  assert!(
    matches!(r, Err(Error::BoundedDecode(_))),
    "buffer larger than the cap must be rejected, got {r:?}"
  );
  assert_eq!(
    out.len(),
    0,
    "rejected reservation must not append anything"
  );
  assert!(
    out.capacity() <= cap,
    "rejected reservation must not allocate past the cap: capacity={}",
    out.capacity()
  );
}

/// Over-cap rejection arithmetic: a corrupt/hostile decoder
/// buffer count can present `n == usize::MAX` (or any value that, summed
/// with `out.len()`, overflows `usize`). The rejection branch must NOT
/// panic in debug or wrap in release while building the diagnostic
/// `observed` field of the BoundedDecode payload — it must return the
/// recoverable error with the saturated observed count.
#[test]
fn reserve_under_cap_rejects_overflowing_n_without_panic() {
  let mut out: Vec<f32> = vec![0.0; 1]; // out.len() > 0 so the sum can overflow
  let cap = 1024usize;
  let n = usize::MAX;
  let r = reserve_under_cap(&mut out, n, cap);
  match r {
    Err(Error::BoundedDecode(p)) => {
      assert_eq!(p.cap(), cap as u64, "cap field must be the configured cap");
      // The diagnostic observed count must be saturated, not wrapped.
      // `out.len() (1) + usize::MAX` overflows usize, so the saturated
      // u64 sum equals u64::MAX (since `usize::MAX as u64` already
      // reaches u64::MAX on 64-bit, and the saturating_add to 1 stays
      // at u64::MAX).
      assert_eq!(
        p.observed(),
        u64::MAX,
        "observed must be saturated at u64::MAX, not wrapped"
      );
    }
    other => panic!("expected BoundedDecode err, got {other:?}"),
  }
  // The Vec must not have grown.
  assert_eq!(out.len(), 1, "rejected reservation must not append");
}

/// An exactly-fitting buffer is accepted and reserves the room (so the
/// caller's subsequent pushes cannot regrow).
#[test]
fn reserve_under_cap_accepts_and_reserves_up_to_cap() {
  let cap = 32usize;
  let mut out: Vec<f32> = Vec::new();
  reserve_under_cap(&mut out, cap, cap).expect("filling exactly to cap must succeed");
  assert!(
    out.capacity() >= cap,
    "reservation did not provide cap room: capacity={} cap={cap}",
    out.capacity()
  );
  // Pushing the reserved `cap` samples cannot reallocate (capacity was
  // reserved), so this loop never hits the infallible-growth path.
  for i in 0..cap {
    out.push(i as f32);
  }
  assert_eq!(out.len(), cap);
}

/// Cap-limited reserve growth: a plain amortized
/// `Vec::try_reserve(n)` can grow capacity to ~2× the *current* capacity
/// even when only a few samples remain under `cap`. With a near-cap
/// header hint (capacity already reserved up to `cap` by
/// `try_reserve_exact` in `load_audio`) plus a final in-cap packet, that
/// doubling would attempt an allocation FAR larger than the cap —
/// defeating the [`MAX_DECODED_SAMPLES`] memory ceiling (and, under
/// memory pressure, spuriously failing an in-cap decode because the
/// oversized reserve fails). The cap-limited growth must (a) accept the
/// in-cap packet and (b) NOT grow capacity past `cap`.
#[test]
fn reserve_under_cap_growth_does_not_exceed_cap() {
  let cap = 64usize;
  // `out` already holds capacity == cap (the near-cap header-hint state)
  // and is filled to one below the cap, so a 1-sample packet still fits
  // under the cap but the *needed* capacity (cap) equals current
  // capacity — the case where a plain `try_reserve` would otherwise
  // double to ~2*cap.
  let mut out: Vec<f32> = Vec::with_capacity(cap);
  out.resize(cap - 1, 0.0);
  assert_eq!(out.capacity(), cap, "precondition: capacity == cap");

  // A final packet that still fits under the cap is accepted.
  reserve_under_cap(&mut out, 1, cap).expect("in-cap final packet must be accepted");

  // The reservation must NOT have doubled capacity past the cap.
  assert!(
    out.capacity() <= cap,
    "reserve grew capacity past the cap: capacity={} cap={cap}",
    out.capacity()
  );

  // Also exercise the headroom case: capacity == cap - packet_len with
  // plenty of slack below the cap. Growth is still clamped at the cap.
  let packet = 8usize;
  let mut out2: Vec<f32> = Vec::with_capacity(cap - packet);
  out2.resize(cap - packet, 0.0);
  let cap2_before = out2.capacity();
  reserve_under_cap(&mut out2, packet, cap).expect("packet filling exactly to cap must succeed");
  assert!(
    out2.capacity() <= cap,
    "reserve grew capacity past the cap: capacity={} cap={cap}",
    out2.capacity()
  );
  assert!(
    out2.capacity() >= cap2_before + packet,
    "reserve did not provide room for the packet: capacity={} need>={}",
    out2.capacity(),
    cap2_before + packet
  );
  // The reserved room is real: pushing the packet cannot reallocate.
  for i in 0..packet {
    out2.push(i as f32);
  }
  assert_eq!(out2.len(), cap);
}

// ---- save_wav atomic-rename durability + xattr preservation
//      (see #135, #138) -------------------------------------------------

/// Unique per-test temp dir under `std::env::temp_dir()`. Process-scoped
/// + test-named so parallel test binaries / cases never collide.
fn audio_temp_dir(name: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("mlxrs_audio_io_{}_{}", std::process::id(), name));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

/// Comment-stripper for the
/// source-structural tests. Removes line comments (`// ... \n`) and
/// non-nested block comments (`/* ... */`) so a probe / fsync ident
/// hidden in commentary cannot satisfy `contains`-style assertions.
/// Rust block comments are nestable but the call sites here only run
/// on hand-written code slices, so the simpler non-nested pass is
/// sufficient — and a future regression that buries a probe inside
/// a nested block comment is no more permissive than the baseline.
/// String-literal contents are preserved verbatim (the test cases
/// explicitly look for `"system.posix_acl_access"` etc. AS strings).
///
/// Iteration is char-by-char (not byte-by-byte) so the non-ASCII
/// characters that appear in `io.rs` doc-comment commentary (em-dashes
/// etc.) round-trip through the stripper without UTF-8 corruption.
fn strip_rust_comments(src: &str) -> String {
  let mut out = String::with_capacity(src.len());
  let mut chars = src.chars().peekable();
  while let Some(c) = chars.next() {
    // Line comment `// ...\n`.
    if c == '/' && chars.peek() == Some(&'/') {
      chars.next();
      for nc in chars.by_ref() {
        if nc == '\n' {
          out.push('\n');
          break;
        }
      }
      continue;
    }
    // Block comment `/* ... */` (non-nested).
    if c == '/' && chars.peek() == Some(&'*') {
      chars.next();
      let mut prev = '\0';
      for nc in chars.by_ref() {
        if prev == '*' && nc == '/' {
          break;
        }
        prev = nc;
      }
      continue;
    }
    // String literal `"..."` — preserve contents verbatim, handle
    // `\\"` and `\\\\` escapes so an embedded `"` doesn't end the
    // literal early.
    if c == '"' {
      out.push('"');
      while let Some(nc) = chars.next() {
        out.push(nc);
        if nc == '\\' {
          if let Some(esc) = chars.next() {
            out.push(esc);
          }
          continue;
        }
        if nc == '"' {
          break;
        }
      }
      continue;
    }
    out.push(c);
  }
  out
}

/// `save_wav` must call `fsync(dirfd)` on the
/// parent directory after the atomic rename, so the directory-entry
/// update is durable on disk (otherwise a crash between rename and
/// writeback can leave the FS with no entry for the new file on
/// ext4/xfs/APFS — the renamed file disappears on power loss).
///
/// We can't directly observe an fsync syscall from a unit test
/// without a syscall tracer, but we CAN assert:
///   1. `save_wav` returns `Ok` (so the fsync didn't spuriously
///      flunk the call on a platform that supports it).
///   2. The file is observable at `path` immediately after the call
///      (so the parent-dir fsync isn't blocking on a stale handle).
///   3. The save still works when the destination has no parent
///      (relative-path-with-no-slashes is the `Path::new(".")`
///      fallback in `fsync_parent_dir` — it must not crash).
///
/// The strictness of (2) is the regression guard: if the implementation
/// silently fails the dir-fsync without propagating, `save_wav` still
/// returns `Ok` and the file is observable, which is the correct
/// best-effort contract documented at the call site.
#[test]
fn save_wav_fsyncs_parent_dir_after_rename() {
  let dir = audio_temp_dir("audio9_fsync_parent");
  let path = dir.join("out.wav");
  let samples = vec![0.0_f32, 0.5, -0.5, 0.25, -0.25, 1.0, -1.0, 0.0];
  save_wav(&path, &samples, 16_000).expect("save_wav must succeed on a fresh path");
  assert!(
    path.exists(),
    "post-save path must be observable (parent-dir fsync did not corrupt the rename)"
  );
  let meta = fs::metadata(&path).expect("destination metadata must be readable");
  // 44-byte WAV header + 16 i16 samples = 44 + 16 = 60 bytes.
  assert_eq!(
    meta.len(),
    44 + 2 * samples.len() as u64,
    "post-save WAV size must match the header + i16 samples body"
  );
  // Overwrite to exercise the existing-destination path (which captures
  // perms/xattrs and runs the same dir-fsync after the rename).
  save_wav(&path, &samples, 16_000).expect("save_wav overwrite must succeed");
  assert!(
    path.exists(),
    "post-overwrite path must still be observable"
  );
}

/// On Unix `save_wav` must preserve the
/// destination's extended attributes (Linux user xattrs, POSIX-1.e
/// ACLs in `system.posix_acl_access`, SELinux contexts, macOS
/// xattrs, etc.) across the atomic-rename. Without preservation, a
/// destination with `chmod +a "user allow read"` or `setfacl -m
/// u:bob:r` would silently lose those entries when overwritten.
///
/// We set a `user.mlxrs.p6_audio10` xattr on a fresh destination,
/// call `save_wav` to overwrite, then verify the xattr survives.
/// Gated `#[cfg(unix)]` because the `xattr` crate is only linked
/// on Unix targets (Cargo.toml `[target.'cfg(unix)'.dependencies]`).
/// Inside the cfg, also gated on `xattr::SUPPORTED_PLATFORM` so
/// Unix-flavored systems whose `target_os` falls outside the
/// crate's supported list (none common today, but future-proof)
/// silently skip. Some environments — notably tmpfs without
/// `user.*` xattr support, or sandboxed CI runners that mount the
/// temp dir with `nosuid,nodev,nouser_xattr` — also reject the
/// initial `xattr::set`; we treat that initial-set failure as a
/// "platform doesn't expose user xattrs here" skip rather than a
/// test failure, so the test stays portable across runners.
#[cfg(unix)]
#[test]
fn save_wav_preserves_xattrs_on_overwrite() {
  if !xattr::SUPPORTED_PLATFORM {
    return; // Unsupported Unix variant — the read returns ENOTSUP.
  }
  let dir = audio_temp_dir("audio10_xattr_preserve");
  let path = dir.join("out.wav");
  let samples = vec![0.0_f32, 0.1, -0.1, 0.0];
  // Create the destination first (so an xattr can be attached before
  // the overwriting save_wav runs).
  save_wav(&path, &samples, 16_000).expect("initial save_wav must succeed");
  // Attach a user-namespace xattr. Skip the test if the filesystem
  // backing `std::env::temp_dir()` does not accept user xattrs (a
  // common sandbox configuration — tmpfs `nouser_xattr`, some CI
  // mounts, etc.); we have no way to express the preservation
  // contract on a platform that has nothing to preserve.
  let xattr_name = "user.mlxrs.p6_audio10";
  let xattr_value: &[u8] = b"p6-audio10-marker";
  if xattr::set(&path, xattr_name, xattr_value).is_err() {
    return; // Backing FS doesn't expose user xattrs — skip.
  }
  // Overwrite via save_wav — the xattr MUST be preserved.
  let new_samples = vec![0.5_f32, -0.5, 0.25, -0.25];
  save_wav(&path, &new_samples, 16_000).expect("overwriting save_wav must succeed");
  // Read back: the xattr must still be there with the same bytes.
  let got = xattr::get(&path, xattr_name).expect("xattr::get on the overwritten file must succeed");
  assert_eq!(
    got.as_deref(),
    Some(xattr_value),
    "xattr {xattr_name:?} was lost during the save_wav overwrite — \
       capture_xattrs/restore_xattrs is not preserving the user namespace"
  );
}

/// `capture_xattrs` on Unix must
/// EXPLICITLY probe known ACL/security xattr names in addition to
/// walking `xattr::list`, because the kernel is allowed to omit
/// `system.*` from `listxattr` (POSIX-1.e ACLs commonly are) and
/// will hide `trusted.*` from non-root callers. Without the
/// explicit-probe path, a destination with a POSIX ACL stored in
/// `system.posix_acl_access` would be silently dropped on overwrite.
///
/// True end-to-end coverage of an ACL/security xattr requires root
/// privileges and a filesystem that exposes the namespace — we have
/// neither in unit tests. The structural guard here verifies the
/// PROBE PATH RUNS even when the explicit-probe attribute is absent
/// from the destination (the more dangerous regression mode is the
/// probe getting removed entirely): we set a representative
/// explicit-probe name from the user namespace (we cannot set
/// `system.posix_acl_access` without ACL machinery, but `user.*`
/// xattrs are accepted on any user-xattr-capable filesystem and the
/// probe loop's structure is identical for any name), call
/// `capture_xattrs` directly, and assert the read succeeded. The
/// matching source-level test below (`..._explicit_probes`) asserts
/// the explicit-probe NAMES are present in the source so a future
/// edit can't silently remove the security/ACL probes without
/// failing the suite.
#[cfg(unix)]
#[test]
fn capture_xattrs_returns_some_on_existing_unix_path() {
  if !xattr::SUPPORTED_PLATFORM {
    return;
  }
  let dir = audio_temp_dir("audio10_xattr_capture");
  let path = dir.join("probe.wav");
  let samples = vec![0.0_f32, 0.0];
  save_wav(&path, &samples, 16_000).expect("save_wav must succeed");
  // The captured set is at least defined (Some) for an existing path
  // on a supported platform — None is reserved for the
  // path-doesn't-exist / listxattr-failed cases.
  let captured = capture_xattrs(&path);
  assert!(
    captured.is_some(),
    "capture_xattrs must return Some on an existing path on a supported platform"
  );
}

/// Source-structural guard. The
/// explicit-probe set inside `capture_xattrs` (Unix arm) must
/// continue to include the ACL/security xattr names — a future
/// refactor that drops these probes would silently regress
/// preservation of POSIX ACLs and SELinux labels.
///
/// **Narrowed scan**: a guard that scanned the whole
/// `io.rs` file with `src.contains(needle)` would pass even
/// if `EXPLICIT_PROBES` were entirely removed — the probe names
/// appear in the xattr-rationale doc-comments above and in this
/// test's own assertion array. Narrow the scan to the slice of
/// source bytes between `const EXPLICIT_PROBES:` and the closing
/// `;`, strip line + block comments from that slice, and assert
/// each probe name appears as a quoted string literal within the
/// non-comment portion.
#[test]
fn capture_xattrs_source_includes_acl_security_explicit_probes() {
  let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/audio/io.rs"))
    .expect("io.rs must be readable from CARGO_MANIFEST_DIR");

  // Find the `const EXPLICIT_PROBES:` declaration and slice it from
  // the start of the declaration to the next `;` (the array literal
  // is single-statement in current code, so the next `;` closes it).
  let decl_start = src
    .find("const EXPLICIT_PROBES:")
    .expect("capture_xattrs must declare a `const EXPLICIT_PROBES:` constant");
  let decl_end_rel = src[decl_start..]
    .find(';')
    .expect("EXPLICIT_PROBES declaration must terminate with `;`");
  let decl_slice = &src[decl_start..decl_start + decl_end_rel];

  // Strip line comments (`// ... \n`) and block comments
  // (`/* ... */`) so a probe name buried in a comment can't satisfy
  // the assertion below. The current EXPLICIT_PROBES literal has no
  // comments interleaved between its quoted strings, but stripping
  // is defense-in-depth against a future edit that adds annotation
  // comments while removing one of the probe names.
  let stripped = strip_rust_comments(decl_slice);

  for needle in [
    "system.posix_acl_access",
    "system.posix_acl_default",
    "security.selinux",
    "security.capability",
    "security.ima",
    "security.evm",
  ] {
    // Match the quoted-string form so the probe must appear as a
    // string literal, not just any source token.
    let quoted = format!("\"{needle}\"");
    assert!(
      stripped.contains(&quoted),
      "capture_xattrs EXPLICIT_PROBES must include {needle:?} as a \
         string literal (the ACL/security namespace `listxattr` may \
         omit) — removing it silently regresses the #138 ACL/security xattr capture.\n\
         Inspected (comments stripped) slice:\n{stripped}"
    );
  }
}

/// Source-structural
/// guard. The `save_wav` body must call the
/// [`save_wav_post_metadata_fsync`] helper AFTER `restore_xattrs`
/// AND BEFORE the `fs::rename` that publishes the tempfile. Without
/// that ordering a crash between metadata restoration and the
/// parent-dir fsync would leave the renamed file with stale
/// permissions/xattrs (the bytes are durable, but the inode
/// metadata isn't). We verify the ordering by scanning the source —
/// we cannot observe an fsync syscall from userspace without a
/// syscall tracer.
///
/// **Helper extraction kills the substring
/// fragility.** A guard that sliced each cfg-branch of an inline
/// `let sync_result = ...` binding and asserted that the
/// `meta_file.sync_all(` token appeared in BOTH would be fragile — even
/// with comment-stripping, a regression that dropped the real call while
/// leaving the token in an error/debug string literal inside the
/// same branch would keep the guard green (string literals survive
/// comment-stripping by design). We factor the post-metadata fsync
/// into a named helper [`save_wav_post_metadata_fsync`] so the call
/// site becomes a single distinctive function call. We assert the
/// token `save_wav_post_metadata_fsync(` appears between
/// `restore_xattrs(` and `fs::rename(` in the function body —
/// a string-literal collision on a name that specific is implausible
/// in calling code (callers don't pass function names as string
/// arguments). The test-only failure-injection branch lives inside
/// the helper, not at the call site, so there is no inline
/// `#[cfg(test)] / #[cfg(not(test))]` split — there is
/// exactly one call site to find.
///
/// Substring-based ordering guards are inherently fragile around
/// comments and string literals; narrowing the substring to a single
/// distinctive helper name that is implausible to collide with
/// non-call source keeps the guard robust.
///
/// PAIRS with the behavioral test
/// [`save_wav_post_metadata_fsync_helper_is_called_before_rename_runtime`]
/// which uses the test-only failure-injection hook to prove the
/// helper IS invoked at runtime (failure-injection propagates as
/// `Err` → no rename → original bytes preserved).
#[test]
fn save_wav_calls_post_metadata_fsync_helper_before_rename() {
  let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/audio/io.rs"))
    .expect("io.rs must be readable from CARGO_MANIFEST_DIR");
  // Narrow to the body of `save_wav` (from `pub fn save_wav` up to
  // the next free-standing `pub fn ` or `fn ` at column 0 — i.e. the
  // next top-level item) so a doc-comment or test using these names
  // elsewhere can't satisfy the ordering check.
  let sig_idx = src
    .find("pub fn save_wav")
    .expect("save_wav function must be defined");
  // The next top-level `\nfn ` or `\npub fn ` after the signature
  // ends the function body — `save_wav` is the only `pub fn` of that
  // name, and helpers below it (e.g. `fn capture_xattrs`,
  // `fn restore_xattrs`, `fn fsync_parent_dir`, `fn open_excl_tempfile`)
  // start at column 0.
  let body_after = &src[sig_idx..];
  let body_end_rel = body_after[1..]
    .find("\nfn ")
    .or_else(|| body_after[1..].find("\npub fn "))
    .expect("save_wav body must be followed by another top-level fn item");
  let body = &body_after[..1 + body_end_rel];
  // Strip comments so commentary mentioning the helper name cannot
  // satisfy the ordering assertion. String literals are NOT stripped
  // (that's a non-trivial lexer), but the helper name
  // `save_wav_post_metadata_fsync` is distinctive enough that
  // calling code is the only plausible source of this token —
  // callers don't pass function names as string arguments.
  let stripped = strip_rust_comments(body);

  let restore_idx = stripped
    .find("restore_xattrs(")
    .expect("save_wav must call restore_xattrs before the rename");
  let rename_idx = stripped[restore_idx..]
    .find("fs::rename(")
    .map(|o| restore_idx + o)
    .expect(
      "save_wav must call fs::rename AFTER restore_xattrs — \
         see #138",
    );
  // The slice between `restore_xattrs(` and `fs::rename(` is where
  // the post-metadata fsync helper call must live.
  let between = &stripped[restore_idx..rename_idx];

  // Assert the distinctive helper-call
  // token appears between `restore_xattrs(` and `fs::rename(`. The
  // call form is `save_wav_post_metadata_fsync(&meta_file)`; we
  // match the leading `save_wav_post_metadata_fsync(` token because
  // a regression that swapped the borrow expression (e.g.
  // `(&mut meta_file)`) should still keep the helper-invocation
  // ordering valid as long as the helper is called.
  let helper_token = "save_wav_post_metadata_fsync(";
  let helper_rel = between.find(helper_token).expect(
    "save_wav must call `save_wav_post_metadata_fsync(` between \
       restore_xattrs and fs::rename — see #138. \
       The helper folds the test-only failure-injection branch + the \
       real `meta_file.sync_all()` call so the call site is a single \
       distinctive function call (no cfg-branch substring matching). \
       A regression that dropped this call would leave the \
       post-metadata fsync unrun on the metadata-restoration path, \
       and the restored permissions/xattrs would not be durable \
       across a crash between rename and parent-dir fsync.",
  );
  let helper_abs = restore_idx + helper_rel;

  // Ordering sanity: helper call must sit strictly between
  // `restore_xattrs(` and `fs::rename(` in absolute index space.
  // The `between` slice was carved from that range, so the find
  // above already satisfies this — we re-assert with absolute
  // indices for a clearer diagnostic.
  assert!(
    restore_idx < helper_abs && helper_abs < rename_idx,
    "save_wav ordering broken: restore_xattrs @ {restore_idx}, \
       save_wav_post_metadata_fsync @ {helper_abs}, fs::rename @ \
       {rename_idx} — the helper call must sit between restore_xattrs \
       and fs::rename"
  );
}

/// **SMOKE TEST** of the
/// chmod-restore path; NOT the behavioral guard for the post-metadata
/// fsync regression. Pre-create the destination with mode `0o444` (no
/// write bit for owner), then call `save_wav` to overwrite. An
/// implementation that captured those perms, restored them onto the
/// tempfile, then REOPENED the tempfile with
/// `OpenOptions::new().write(true)` for the post-metadata fsync would
/// be unsafe — that reopen could fail with EACCES (the process owns the
/// inode so the chmod succeeded, but the inode mode now lacks the owner
/// write bit so a subsequent write-open is rejected), and an
/// `if let Ok(..)` would silently swallow the EACCES — letting the
/// rename publish a file whose chmod/xattrs were not yet on stable
/// storage. The code keeps the ORIGINAL writable handle alive from
/// tempfile creation and `sync_all`s on that handle, so the inode
/// mode bits don't block the metadata flush.
///
/// **Regression-class clarification.**
/// Under the reopen-EACCES bug the perms would be restored *before* the
/// buggy reopen, the reopen EACCES would be swallowed, and the rename
/// would still proceed — meaning the assertions below (Ok return, new
/// bytes observable, mode restored) would ALL still pass. This test
/// cannot distinguish that bug from the correct path on its own; the
/// behavioral
/// guard for the post-metadata fsync regression is
/// [`save_wav_post_metadata_fsync_helper_is_called_before_rename_runtime`]
/// which uses the test-only [`set_force_meta_fsync_failure`] hook to
/// inject a failure at the fsync site and assert the cleanup
/// (tempfile gone, original bytes preserved, `Error::FileIo`
/// returned). This test STAYS as an end-to-end smoke test of the
/// chmod-restore + rename path, and PAIRS with the source-structural
/// test `save_wav_calls_post_metadata_fsync_helper_before_rename`
/// (which asserts the distinctive helper-call token
/// `save_wav_post_metadata_fsync(` appears between `restore_xattrs(`
/// and `fs::rename(` in the function body — a later refactor
/// replaced the prior `meta_file.sync_all(` token scan to
/// kill string-literal false positives). The three together cover
/// the regression class: smoke (end-to-end chmod-restore),
/// structural (helper-call ordering preserved across refactors), and
/// behavioral (failure-injection proves the error-propagation arm).
///
/// Observable signals this test asserts:
///  1. `save_wav` returns `Ok` (it must not fail under read-only
///     overwrite).
///  2. The published file contains the NEW bytes (not the initial
///     bytes) — proves the rename proceeded.
///  3. The captured `0o444` mode is restored on the published file
///     — proves `set_permissions` ran AND was not undone.
///
/// Xattr restoration on a `0o444` inode is OUT OF SCOPE: on
/// macOS+APFS (and on Linux+ext4 without the `user_xattr` mount
/// option) `xattr::set` on a 0o444 file returns EACCES, which
/// `restore_xattrs` silently swallows by design (per-xattr failures
/// must not poison the save — see `restore_xattrs` rationale). The
/// xattr-preservation contract is covered by the sibling test
/// `save_wav_preserves_xattrs_on_overwrite` which exercises a
/// writable destination.
#[cfg(unix)]
#[test]
fn save_wav_read_only_overwrite_fsyncs_metadata_before_rename() {
  use std::os::unix::fs::PermissionsExt;
  let dir = audio_temp_dir("audio138_ro_overwrite");
  let path = dir.join("ro.wav");

  // Pre-create the destination with some initial bytes, then flip
  // to 0o444. We MUST create-then-chmod (rather than create at 0o444
  // directly) so that the initial `save_wav` itself is unaffected
  // — the failure-mode under test is the SECOND call (the overwrite).
  let initial = vec![0.0_f32, 0.0];
  save_wav(&path, &initial, 16_000).expect("initial save_wav must succeed");
  fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444))
    .expect("set_permissions(0o444) on pre-existing file must succeed");

  // Now overwrite via save_wav. The writable handle
  // is kept alive past `set_permissions(tmp, 0o444)`, so the
  // post-metadata sync_all does NOT need to reopen-for-write —
  // EACCES on the inode is irrelevant for an already-open handle.
  // The reopen-EACCES bug would STILL return Ok (it is a
  // durability bug, not a visibility one), so signal #1 alone is a
  // smoke-only regression guard; signal #3 (mode restored) is the
  // proof that the perms-restore-then-fsync ordering executed end
  // to end on this code path.
  let new_samples: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) / 32.0).collect();
  save_wav(&path, &new_samples, 16_000)
    .expect("save_wav overwrite of 0o444 destination must succeed (#138)");

  // Signal #1 + #2: the new bytes must be observable at the
  // published path.
  assert!(path.exists(), "post-overwrite path must be observable");
  let meta = fs::metadata(&path).expect("destination metadata must be readable");
  assert_eq!(
    meta.len(),
    44 + 2 * new_samples.len() as u64,
    "post-overwrite WAV size must reflect the NEW sample buffer (not the initial buffer)"
  );

  // Signal #3: the pre-existing 0o444 mode bits must be restored on
  // the published file — proving `set_permissions` ran AND
  // `fs::rename` proceeded AFTER the metadata-restoration block.
  // A regression that skipped restore_xattrs /
  // set_permissions would leave the file at the tempfile's umask
  // mode (typically 0o644).
  let mode = meta.permissions().mode() & 0o777;
  assert_eq!(
    mode, 0o444,
    "captured 0o444 mode must be restored on the published file \
       (got {mode:#o}); the post-metadata fsync must run on the \
       ORIGINAL writable handle so it doesn't fail with EACCES on \
       reopen — see #138"
  );

  // Restore writable mode so audio_temp_dir's `remove_dir_all` on the
  // next run can unlink the file.
  let _ = fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644));
}

/// **BEHAVIORAL guard** for the
/// post-metadata fsync regression, and the RUNTIME counterpart to
/// the structural test
/// [`save_wav_calls_post_metadata_fsync_helper_before_rename`].
/// Uses the test-only [`set_force_meta_fsync_failure`] hook to force
/// the [`save_wav_post_metadata_fsync`] helper to return an injected
/// `io::Error`, then asserts the error-propagation arm:
///
///  1. `save_wav` returns `Err(Error::FileIo(_))` (op = `Fsync`) whose
///     context mentions `sync_all` and whose path references the
///     tempfile — proving the failure was NOT silently swallowed
///     and the post-metadata fsync helper was actually
///     invoked.
///  2. The destination at `path` still contains the ORIGINAL bytes
///     (the rename was NOT performed) — proving the failure path
///     short-circuits before `fs::rename`.
///  3. The staging directory contains NO `<basename>.<pid>.<rand>.tmp`
///     leftovers — proving `fs::remove_file(&tmp_path)` ran on the
///     error path (operator visibility hygiene).
///
/// This is the actual
/// behavioral guard for the durability bug — the sibling smoke test
/// `save_wav_read_only_overwrite_fsyncs_metadata_before_rename`
/// cannot distinguish the reopen-EACCES bug from the correct path on its
/// own (perms would be restored before the buggy reopen and the EACCES
/// swallowed, so the smoke test's three signals would all still pass).
/// Failure injection on a writable destination is the only way to
/// behaviorally observe the fsync-error arm from userspace without a
/// crash-and-restart harness.
///
/// **Runtime helper-invocation contract.**
/// After factoring the post-metadata fsync into
/// [`save_wav_post_metadata_fsync`], this test simultaneously proves
/// the helper IS called at runtime (signal #1 — only the helper's
/// `cfg(test)` branch produces the injected error string, so a
/// regression that dropped the helper call from `save_wav` would
/// silently `Ok(())` the fsync arm and signal #2 would fail because
/// the rename WOULD proceed) and that its failure is correctly
/// propagated (signal #2 + #3). The structural sibling guards the
/// CALL ORDER in source; this test guards the CALL ITSELF + the
/// failure-propagation contract at runtime.
///
/// Flag-reset discipline: the test sets the flag, then RAII-resets
/// it in a `Drop` guard so a subsequent test scheduled by the harness
/// on the SAME WORKER THREAD doesn't observe an injected failure
/// even if this test panics. **Thread-local scoping**: the flag
/// is now thread-local (`thread_local!` `Cell<bool>`), so concurrent
/// tests on other worker threads cannot be poisoned by the
/// injection — they observe the default `false` regardless of
/// `--test-threads` setting. The drop-guard reset still matters
/// because the harness reuses worker threads for sequential tests
/// within a binary, and we don't want a leaked `true` to make the
/// next test on this same worker observe an injected fsync failure.
#[test]
fn save_wav_post_metadata_fsync_helper_is_called_before_rename_runtime() {
  // Drop-guard reset: even on panic, the flag is cleared before the
  // next test runs. `Drop` is invoked on unwind, so this is safer
  // than a plain `set_force_meta_fsync_failure(false)` at the end of
  // the body (which would be skipped on assertion failure).
  struct ResetOnDrop;
  impl Drop for ResetOnDrop {
    fn drop(&mut self) {
      set_force_meta_fsync_failure(false);
    }
  }

  let dir = audio_temp_dir("audio138_fsync_failure");
  let path = dir.join("dest.wav");

  // Pre-create the destination with known bytes (a valid WAV so
  // `existing_perms`/`existing_xattrs` capture succeeds and the
  // post-metadata fsync arm is entered — the arm is GATED on
  // `existing_perms.is_some() || existing_xattrs.is_some()`).
  let initial_samples = vec![0.5_f32; 256];
  save_wav(&path, &initial_samples, 16_000).expect("initial save_wav must succeed");
  let original_bytes = fs::read(&path).expect("must read initial bytes");
  assert!(
    !original_bytes.is_empty(),
    "initial save must produce non-empty bytes"
  );

  // Arm the failure injection AFTER the initial save so the initial
  // save isn't affected.
  set_force_meta_fsync_failure(true);
  let _reset = ResetOnDrop; // arm the RAII reset.

  // Overwrite attempt — must hit the injected fsync failure.
  let new_samples: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) / 512.0).collect();
  let result = save_wav(&path, &new_samples, 16_000);

  // Signal #1: Err(Error::FileIo) whose context mentions the
  // tempfile path AND the `sync_all` failure mode (op=Fsync). We don't bind
  // to the EXACT injected-error string (that's a test-implementation
  // detail) — instead we assert the user-facing payload reflects the
  // failure class.
  match &result {
    Err(Error::FileIo(payload)) => {
      assert_eq!(
        payload.op(),
        FileOp::Fsync,
        "post-metadata fsync error must carry FileOp::Fsync"
      );
      assert!(
        payload.context().contains("sync_all"),
        "post-metadata fsync error context must mention sync_all; got {}",
        payload.context()
      );
      let path_str = payload.path().display().to_string();
      assert!(
        path_str.contains(".tmp"),
        "post-metadata fsync error path must reference the tempfile; got {path_str}"
      );
    }
    other => {
      panic!("save_wav must return Err(Error::FileIo) on injected fsync failure; got {other:?}")
    }
  }

  // Signal #2: the destination MUST still contain the ORIGINAL
  // bytes — proves `fs::rename` was NEVER called on the failure
  // path (the no-rename-on-failure guarantee).
  let post_bytes =
    fs::read(&path).expect("destination must still exist (rename was not attempted)");
  assert_eq!(
    post_bytes, original_bytes,
    "destination bytes must be UNCHANGED on injected fsync failure; \
       rename must NOT proceed when the post-metadata sync_all fails"
  );

  // Signal #3: NO `*.tmp.*`-style tempfile under the staging dir —
  // proves `fs::remove_file(&tmp_path)` cleanup ran on the error
  // path (operator visibility hygiene). The tempfile naming pattern
  // is `<basename>.<pid>.<rand>.tmp` (see `open_excl_tempfile`); we
  // scan for any directory entry containing `.tmp` and assert none
  // remain. The published `dest.wav` itself never carries `.tmp` in
  // its name.
  let leftovers: Vec<String> = fs::read_dir(&dir)
    .expect("staging dir must be listable")
    .filter_map(|e| e.ok())
    .map(|e| e.file_name().to_string_lossy().into_owned())
    .filter(|n| n.contains(".tmp"))
    .collect();
  assert!(
    leftovers.is_empty(),
    "staging dir must contain NO tempfile leftovers on injected fsync \
       failure path (operator hygiene); found: {leftovers:?}"
  );
}

/// File-level round-trip regression. Save a
/// known sample, decode it back via `load_audio`, and assert
/// bit-perfect reconstruction for in-range samples (the symmetric
/// `read = / 32768.0` + `write = * 32768.0` convention guarantees
/// `(f * 32768).round() / 32768.0 == f` for every multiple of
/// `1/32768` in `[-1.0, 1.0)`).
///
/// This complements the kernel-level
/// `quantize_read_write_round_trip_is_symmetric` test in
/// `simd::audio::quantize` by exercising the FULL `save_wav` →
/// `load_audio` pipeline (header build, atomic rename, symphonia
/// decode, push_samples normalization). A regression in `I16_MUL`
/// or `I16_DIV` would surface here as bit-drift even when the
/// kernel test passes.
#[test]
fn save_wav_then_load_audio_round_trip_is_bit_exact() {
  let dir = audio_temp_dir("audio4_roundtrip");
  let path = dir.join("rt.wav");
  // A spread of in-range samples representable as exact i16 codepoints
  // under the 32768 scale: f = k / 32768 for various k in
  // [-32768, 32768). Exact in f32.
  let samples: Vec<f32> = [-32_768_i32, -1, 0, 1, 16_384, -16_384, 32_767]
    .iter()
    .map(|&k| (k as f32) / 32_768.0)
    .collect();
  save_wav(&path, &samples, 16_000).expect("save_wav round-trip must succeed");
  let (decoded, sr) = load_audio(&path).expect("load_audio must round-trip the saved WAV");
  assert_eq!(sr, 16_000, "sample rate round-trip mismatch");
  assert_eq!(
    decoded.len(),
    samples.len(),
    "sample count round-trip mismatch"
  );
  for (i, (&orig, &got)) in samples.iter().zip(decoded.iter()).enumerate() {
    assert_eq!(
      got.to_bits(),
      orig.to_bits(),
      "round-trip drift at index {i}: original {orig}, decoded {got}"
    );
  }
}

// ---- resample_linear: closed-form oracles + cap / error branches -------
//
// The scalar reference (mirrored by the NEON kernel) computes, for each
// output index `i`:
//   x      = i * (from_rate / to_rate)
//   lo     = floor(x)
//   frac   = x - lo
//   lo_idx = min(lo, last_in)
//   hi_idx = min(lo_idx + 1, last_in)
//   out[i] = samples[lo_idx] + (samples[hi_idx] - samples[lo_idx]) * frac
// out_len = samples.len() * to_rate / from_rate (integer division, u64
// intermediate). All oracle inputs below pick ratios that are exact in
// binary floating point (integer ratios / powers of two) so the expected
// f32 values are bit-exact — the oracle is hand-computed, never produced
// by calling the function under test.

/// `from_rate == 0` is an invariant violation — rejected before any
/// allocation. (io.rs:1984-1988)
#[test]
fn resample_linear_zero_from_rate_is_invariant_violation() {
  let r = resample_linear(&[0.0, 1.0], 0, 16_000);
  assert!(
    matches!(r, Err(Error::InvariantViolation(_))),
    "from_rate==0 must be InvariantViolation, got {r:?}"
  );
}

/// `to_rate == 0` is an invariant violation. (io.rs:1990-1994)
#[test]
fn resample_linear_zero_to_rate_is_invariant_violation() {
  let r = resample_linear(&[0.0, 1.0], 16_000, 0);
  assert!(
    matches!(r, Err(Error::InvariantViolation(_))),
    "to_rate==0 must be InvariantViolation, got {r:?}"
  );
}

/// Empty input short-circuits to an empty Vec regardless of the rates
/// (the empty check precedes the from==to and resampling paths).
/// (io.rs:1996-1998)
#[test]
fn resample_linear_empty_input_returns_empty() {
  let out = resample_linear(&[], 44_100, 16_000).expect("empty input must be Ok");
  assert!(out.is_empty(), "empty input must yield empty output");
  // Empty must also short-circuit before the zero-rate guards are even
  // relevant for a non-zero rate pair, and before the verbatim path.
  let out2 = resample_linear(&[], 16_000, 16_000).expect("empty input (equal rates) must be Ok");
  assert!(out2.is_empty());
}

/// `from_rate == to_rate` returns a verbatim bit-exact copy through the
/// fallible-reservation path (no FP interpolation drift). (io.rs:1999-2024)
#[test]
fn resample_linear_equal_rates_is_verbatim_copy() {
  let samples = vec![-1.0_f32, -0.5, 0.0, 0.25, 0.75, 1.0];
  let out = resample_linear(&samples, 22_050, 22_050).expect("equal-rate resample must succeed");
  assert_eq!(out.len(), samples.len(), "verbatim copy length mismatch");
  for (i, (&a, &b)) in samples.iter().zip(out.iter()).enumerate() {
    assert_eq!(
      a.to_bits(),
      b.to_bits(),
      "verbatim copy must be bit-exact at index {i}: {a} != {b}"
    );
  }
}

/// Integer downsample 2:1 picks the even-indexed source samples exactly
/// (frac == 0 at every output index). ratio = from/to = 2.0, so
/// out[i] = samples[2*i]. (io.rs:2026-2092)
#[test]
fn resample_linear_downsample_2_to_1_picks_even_indices() {
  let samples = vec![0.0_f32, 0.1, 0.2, 0.3, 0.4, 0.5];
  let out = resample_linear(&samples, 2, 1).expect("2:1 downsample must succeed");
  // out_len = 6 * 1 / 2 = 3.
  assert_eq!(out.len(), 3, "2:1 downsample output length");
  // i=0 -> x=0 -> s[0]; i=1 -> x=2 -> s[2]; i=2 -> x=4 -> s[4].
  let expected = [samples[0], samples[2], samples[4]];
  for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
    assert_eq!(
      got.to_bits(),
      exp.to_bits(),
      "2:1 downsample mismatch at {i}: got {got}, expected {exp}"
    );
  }
}

/// Integer upsample 1:2 with ratio = 0.5 interpolates at the half-sample
/// positions (frac == 0.5, exact in f32). For samples = [0.0, 1.0]:
///   i=0 x=0.0 -> 0.0
///   i=1 x=0.5 -> 0.0 + (1.0-0.0)*0.5 = 0.5
///   i=2 x=1.0 -> s[1] = 1.0
///   i=3 x=1.5 -> lo_idx=min(1,1)=1, hi_idx=1 -> 1.0 (clamped tail)
/// (io.rs:2026-2092)
#[test]
fn resample_linear_upsample_1_to_2_interpolates_halves() {
  let samples = vec![0.0_f32, 1.0];
  let out = resample_linear(&samples, 1, 2).expect("1:2 upsample must succeed");
  // out_len = 2 * 2 / 1 = 4.
  let expected = [0.0_f32, 0.5, 1.0, 1.0];
  assert_eq!(out.len(), expected.len(), "1:2 upsample output length");
  for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
    assert_eq!(
      got.to_bits(),
      exp.to_bits(),
      "1:2 upsample mismatch at {i}: got {got}, expected {exp}"
    );
  }
}

/// A downsample ratio so aggressive that `in_len * to_rate / from_rate`
/// floors to 0 returns an empty Vec via the `out_len == 0` early return
/// (NOT the empty-input or verbatim paths). (io.rs:2046-2048)
#[test]
fn resample_linear_out_len_zero_returns_empty() {
  // in_len = 1, to_rate = 1, from_rate = 4 -> out_len = 1*1/4 = 0.
  let out = resample_linear(&[0.5], 4, 1).expect("zero-output-length resample must be Ok");
  assert!(
    out.is_empty(),
    "an output length that floors to 0 must yield an empty Vec, got len {}",
    out.len()
  );
}

/// An adversarial `from_rate=1, to_rate=u32::MAX` ratio drives the output
/// length above [`MAX_RESAMPLED_SAMPLES`] — rejected with `CapExceeded`
/// BEFORE any reservation (the cap check precedes `try_reserve_exact`), so
/// the test allocates nothing large. (io.rs:2054-2062)
#[test]
fn resample_linear_over_cap_output_is_rejected() {
  // out_len_u64 = 2 * u32::MAX / 1 = 8_589_934_590, fits in usize on
  // 64-bit and far exceeds MAX_RESAMPLED_SAMPLES (64 Mi), so the cap
  // guard fires before allocation.
  let samples = [0.0_f32, 0.0];
  let r = resample_linear(&samples, 1, u32::MAX);
  match r {
    Err(Error::CapExceeded(p)) => {
      assert_eq!(
        p.cap(),
        MAX_RESAMPLED_SAMPLES as u64,
        "cap field must be MAX_RESAMPLED_SAMPLES"
      );
      let expected_out_len = samples.len() as u64 * u64::from(u32::MAX); // / from_rate(1)
      assert_eq!(
        p.observed(),
        expected_out_len,
        "observed must be the (uncapped) computed output length"
      );
      assert!(
        p.observed() > p.cap(),
        "observed must exceed the cap by construction"
      );
    }
    other => panic!("expected CapExceeded for over-cap output length, got {other:?}"),
  }
}

// ---- load_audio_with_max_seconds: up-front duration validation --------
//
// The public entry validates `max_seconds` is a finite value > 0 BEFORE
// any file IO; NaN, +/-inf, zero, and negatives surface as OutOfRange
// without touching the filesystem (so a non-existent path is fine — the
// validation fires first). (io.rs:281-287)

/// All of {NaN, +inf, -inf, 0.0, -0.0, negative} reject up front with
/// `OutOfRange` and never reach the (absent) file.
#[test]
fn load_audio_with_max_seconds_rejects_non_positive_or_non_finite() {
  let nonexistent = Path::new("/mlxrs_no_such_audio_file_for_max_seconds_test.wav");
  for bad in [
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    0.0_f32,
    -0.0_f32,
    -1.5_f32,
  ] {
    let r = load_audio_with_max_seconds(nonexistent, bad);
    assert!(
      matches!(r, Err(Error::OutOfRange(_))),
      "max_seconds={bad} must be rejected up front with OutOfRange (before file IO), got {r:?}"
    );
  }
}

// ---- load_audio: open + probe error branches --------------------------

/// Opening a path that does not exist surfaces `Error::FileIo` with
/// `FileOp::Open` (the very first step of the unified worker).
/// (io.rs:390-397)
#[test]
fn load_audio_missing_file_is_fileio_open_error() {
  let path = Path::new("/mlxrs_definitely_missing_audio_file.wav");
  let r = load_audio(path);
  match r {
    Err(Error::FileIo(p)) => {
      assert_eq!(
        p.op(),
        FileOp::Open,
        "missing-file error must carry FileOp::Open"
      );
    }
    other => panic!("expected FileIo(Open) for a missing file, got {other:?}"),
  }
}

/// A file whose bytes are not a recognizable audio container fails the
/// symphonia probe and surfaces `Error::Parse`. (io.rs:416-423)
#[test]
fn load_audio_unrecognized_bytes_is_parse_error() {
  let dir = audio_temp_dir("audio_probe_garbage");
  let path = dir.join("garbage.wav");
  // 64 bytes of non-container junk (no RIFF/ID3/OggS/fLaC magic). The
  // `.wav` extension is only a hint; symphonia probes the content and
  // finds no supported container.
  fs::write(&path, vec![0x7Au8; 64]).expect("write junk file");
  let r = load_audio(&path);
  assert!(
    matches!(r, Err(Error::Parse(_))),
    "unrecognized container bytes must surface Error::Parse, got {r:?}"
  );
}

// ---- multi-channel rejection: load_audio is mono-only -----------------

/// Build a minimal, fully-valid PCM-16 WAV byte buffer with `channels`
/// channels at `sample_rate`, carrying `frames` interleaved zero frames.
/// Mirrors the 44-byte RIFF/WAVE/fmt/data header `save_wav` emits, but
/// parameterized on channel count so the multi-channel rejection path can
/// be exercised. (Independent of the code under test — a hand-built
/// header, not a `save_wav` call.)
fn build_pcm16_wav(channels: u16, sample_rate: u32, frames: u32) -> Vec<u8> {
  const BITS: u16 = 16;
  let block_align: u16 = channels * (BITS / 8);
  let data_size: u32 = frames * u32::from(block_align);
  let byte_rate: u32 = sample_rate * u32::from(block_align);
  let file_size_minus_8: u32 = 36 + data_size;
  let mut v = Vec::with_capacity(44 + data_size as usize);
  v.extend_from_slice(b"RIFF");
  v.extend_from_slice(&file_size_minus_8.to_le_bytes());
  v.extend_from_slice(b"WAVE");
  v.extend_from_slice(b"fmt ");
  v.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
  v.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
  v.extend_from_slice(&channels.to_le_bytes());
  v.extend_from_slice(&sample_rate.to_le_bytes());
  v.extend_from_slice(&byte_rate.to_le_bytes());
  v.extend_from_slice(&block_align.to_le_bytes());
  v.extend_from_slice(&BITS.to_le_bytes());
  v.extend_from_slice(b"data");
  v.extend_from_slice(&data_size.to_le_bytes());
  v.extend(std::iter::repeat_n(0u8, data_size as usize));
  v
}

/// A 2-channel (stereo) WAV is rejected with `Error::OutOfRange` — the
/// mono-only `load_audio` contract. This exercises the full probe ->
/// default_track -> codec_params -> audio() -> channel-count resolution
/// path before the non-mono rejection fires. (io.rs:468-507)
#[test]
fn load_audio_rejects_stereo_with_out_of_range() {
  let dir = audio_temp_dir("audio_stereo_reject");
  let path = dir.join("stereo.wav");
  // 2 channels, 8 frames -> 16 interleaved i16 samples, all silence.
  fs::write(&path, build_pcm16_wav(2, 16_000, 8)).expect("write stereo WAV");
  let r = load_audio(&path);
  assert!(
    matches!(r, Err(Error::OutOfRange(_))),
    "stereo (2-channel) input must be rejected with OutOfRange, got {r:?}"
  );
}

/// A mono WAV built by the SAME independent header builder decodes
/// successfully to the right sample count — pins the builder as a faithful
/// oracle (so the stereo rejection above isn't a builder artifact) and
/// covers the mono happy path through the hand-built header.
#[test]
fn load_audio_accepts_mono_built_wav() {
  let dir = audio_temp_dir("audio_mono_builder");
  let path = dir.join("mono.wav");
  let frames = 10u32;
  fs::write(&path, build_pcm16_wav(1, 8_000, frames)).expect("write mono WAV");
  let (decoded, sr) = load_audio(&path).expect("mono WAV must decode");
  assert_eq!(sr, 8_000, "sample-rate mismatch");
  assert_eq!(decoded.len(), frames as usize, "mono frame count mismatch");
  // All-silence body decodes to exact 0.0 (PCM 0 / divisor == 0.0).
  assert!(
    decoded.iter().all(|&s| s == 0.0),
    "silence body must decode to all-zero samples"
  );
}

// ---- save_wav / save_wav_into: up-front parameter validation ----------

/// `sample_rate == 0` is rejected with `InvariantViolation` before any
/// tempfile is created. (io.rs:1278-1283)
#[test]
fn save_wav_zero_sample_rate_is_invariant_violation() {
  let dir = audio_temp_dir("audio_zero_sr");
  let path = dir.join("zero_sr.wav");
  let r = save_wav(&path, &[0.0, 0.1], 0);
  assert!(
    matches!(r, Err(Error::InvariantViolation(_))),
    "sample_rate==0 must be InvariantViolation, got {r:?}"
  );
  assert!(
    !path.exists(),
    "no file must be created on the zero-sample-rate validation failure"
  );
}

/// A `sample_rate` above the byte-rate u32 ceiling (`u32::MAX / 2`) is
/// rejected with `OutOfRange` before any tempfile is created. (io.rs:1309-1315)
#[test]
fn save_wav_oversized_sample_rate_is_out_of_range() {
  let dir = audio_temp_dir("audio_big_sr");
  let path = dir.join("big_sr.wav");
  // u32::MAX/2 == 2147483647 is the max; +1 overflows the byte_rate cap.
  let r = save_wav(&path, &[0.0, 0.1], (u32::MAX / 2) + 1);
  assert!(
    matches!(r, Err(Error::OutOfRange(_))),
    "sample_rate past the byte-rate ceiling must be OutOfRange, got {r:?}"
  );
  assert!(
    !path.exists(),
    "no file must be created on the oversized-sample-rate validation failure"
  );
}

/// A non-finite sample (NaN) is rejected UPFRONT with a
/// `LayerKeyed`-wrapped `NonFiniteScalar` (naming the offending index)
/// before any tempfile is created. (io.rs:1321-1331)
#[test]
fn save_wav_non_finite_sample_is_rejected_upfront() {
  let dir = audio_temp_dir("audio_nan_sample");
  let path = dir.join("nan.wav");
  let samples = [0.0_f32, 0.5, f32::NAN, 0.25];
  let r = save_wav(&path, &samples, 16_000);
  assert!(
    matches!(r, Err(Error::LayerKeyed(_))),
    "a NaN sample must be rejected with a LayerKeyed(NonFiniteScalar) error, got {r:?}"
  );
  assert!(
    !path.exists(),
    "no file must be created when an input sample is non-finite"
  );
}

/// `save_wav` into a destination path that is an EXISTING DIRECTORY fails
/// at the publishing rename (`fs::rename(tempfile, dir)` cannot replace a
/// non-empty directory with a file) and surfaces `Error::FileIo` with
/// `FileOp::Rename`. The earlier write+fsync of the tempfile all succeed;
/// only the rename fails. The leftover tempfile is cleaned up on the error
/// path. (io.rs:1716-1724)
#[test]
fn save_wav_rename_onto_existing_directory_is_fileio_rename_error() {
  let dir = audio_temp_dir("audio_rename_onto_dir");
  // The destination "path" is itself a directory containing a child, so a
  // file->dir rename is rejected by the OS (ENOTDIR / EISDIR / ENOTEMPTY
  // depending on platform). It lives under `dir` so the tempfile (created
  // in the same parent) shares the filesystem.
  let dest_dir = dir.join("dest_is_a_dir.wav");
  fs::create_dir_all(&dest_dir).expect("create destination directory");
  fs::write(dest_dir.join("occupant.txt"), b"x").expect("populate destination dir");

  let r = save_wav(&dest_dir, &[0.0, 0.1, -0.1], 16_000);
  match r {
    Err(Error::FileIo(p)) => {
      assert_eq!(
        p.op(),
        FileOp::Rename,
        "rename-onto-directory failure must carry FileOp::Rename, got {:?}",
        p.op()
      );
    }
    other => {
      panic!("expected FileIo(Rename) when renaming onto a non-empty directory, got {other:?}")
    }
  }
  // The error-path cleanup must remove the staging tempfile (no `.tmp`
  // leftovers in the parent staging dir).
  let leftovers: Vec<String> = fs::read_dir(&dir)
    .expect("staging dir listable")
    .filter_map(|e| e.ok())
    .map(|e| e.file_name().to_string_lossy().into_owned())
    .filter(|n| n.contains(".tmp"))
    .collect();
  assert!(
    leftovers.is_empty(),
    "rename-failure path must clean up the tempfile; found leftovers: {leftovers:?}"
  );
}

// ---- open_excl_tempfile: path-shape + retry-exhaustion errors ---------

/// A destination path with no `file_name` component (the filesystem root
/// `/`) is rejected with `OutOfRange` — `open_excl_tempfile` cannot derive
/// a tempfile name. (io.rs:1908-1914)
#[test]
fn open_excl_tempfile_no_file_name_is_out_of_range() {
  // `Path::new("/").file_name()` is None.
  let r = open_excl_tempfile(Path::new("/"), 16);
  assert!(
    matches!(r, Err(Error::OutOfRange(_))),
    "a path with no file_name component must be OutOfRange, got {r:?}"
  );
}

/// A `max_retries == 0` budget never opens any tempfile and immediately
/// exhausts the retry loop, returning `Error::FileIo` with `FileOp::Create`
/// (the retry-exhausted arm — the loop body never executes, so the
/// `last_err.unwrap_or_else(...)` synthetic AlreadyExists error path is
/// taken). (io.rs:1919, 1947-1958)
#[test]
fn open_excl_tempfile_zero_retries_exhausts_budget() {
  let dir = audio_temp_dir("audio_tempfile_zero_retries");
  let dest = dir.join("dest.wav");
  let r = open_excl_tempfile(&dest, 0);
  match r {
    Err(Error::FileIo(p)) => {
      assert_eq!(
        p.op(),
        FileOp::Create,
        "exhausted retry budget must carry FileOp::Create"
      );
    }
    other => panic!("expected FileIo(Create) on a zero-retry budget, got {other:?}"),
  }
}

/// A successful `open_excl_tempfile` returns a path in the SAME parent
/// directory as the destination (so the later rename is single-fs) with
/// the `<file_name>.<pid>.<rand>.tmp` shape, and the file exists on disk.
/// Covers the success arm (io.rs:1932) + tempfile-name construction.
#[test]
fn open_excl_tempfile_success_creates_sibling_tmp() {
  let dir = audio_temp_dir("audio_tempfile_success");
  let dest = dir.join("dest.wav");
  let (tmp_path, file) = open_excl_tempfile(&dest, 4).expect("tempfile open must succeed");
  drop(file);
  assert_eq!(
    tmp_path.parent(),
    Some(dir.as_path()),
    "tempfile must be a sibling of the destination (same parent dir)"
  );
  assert!(
    tmp_path.exists(),
    "tempfile must exist on disk after a successful open"
  );
  let name = tmp_path.file_name().unwrap().to_string_lossy().into_owned();
  assert!(
    name.starts_with("dest.wav.") && name.ends_with(".tmp"),
    "tempfile name must be `<file_name>.<pid>.<rand>.tmp`, got {name}"
  );
  let _ = fs::remove_file(&tmp_path);
}

// ---- fsync_parent_dir: parent-resolution branches (best-effort) -------

/// `fsync_parent_dir` is best-effort and must NOT panic on any of its
/// parent-resolution branches:
///  - a root path `/` whose `.parent()` is `None` (early return),
///  - a bare relative file name whose `.parent()` is `Some("")` (the
///    `Path::new(".")` current-directory fallback),
///  - a normal existing path (the real `open(parent) + sync_all` arm).
///
/// (io.rs:1867-1880)
#[test]
fn fsync_parent_dir_handles_all_parent_branches_without_panic() {
  // No-parent branch: `/`.parent() == None -> early return.
  fsync_parent_dir(Path::new("/"));
  // Empty-parent branch: a bare file name -> parent is "" -> open ".".
  fsync_parent_dir(Path::new("relative_name_no_slash.wav"));
  // Real-parent branch: an actual file under a temp dir.
  let dir = audio_temp_dir("audio_fsync_parent_branches");
  let path = dir.join("present.wav");
  save_wav(&path, &[0.0, 0.0], 16_000).expect("seed file for the real-parent fsync branch");
  fsync_parent_dir(&path);
  // Reaching here without a panic is the assertion.
}

// ---- multi-bit-depth WAV decode: push_samples integer/float arms -------
//
// The existing round-trip test only exercises the 16-bit PCM path (the
// `GenericAudioBufferRef::S16` arm). The builders below construct
// fully-valid RIFF/WAVE files at OTHER bit depths so the remaining
// `push_samples` arms and `int_divisor` widths are decoded through the
// real `load_audio` → symphonia → `push_samples` pipeline.
//
// Format-tag / bit-depth → codec → `GenericAudioBufferRef` variant
// (verified against symphonia-format-riff 0.6 `read_pcm_fmt` /
// `read_ieee_fmt` + symphonia-codec-pcm 0.6 `try_new`):
//   PCM   (0x0001)  8-bit  → CODEC_ID_PCM_U8    → U8  arm
//   PCM   (0x0001) 24-bit  → CODEC_ID_PCM_S24LE → S24 arm (int_divisor(24))
//   PCM   (0x0001) 32-bit  → CODEC_ID_PCM_S32LE → S32 arm (int_divisor(32))
//   IEEE  (0x0003) 32-bit  → CODEC_ID_PCM_F32LE → F32 arm (check_finite)
//   IEEE  (0x0003) 64-bit  → CODEC_ID_PCM_F64LE → F64 arm
//
// The U16/U24/U32/S8 arms are NOT reachable through a WAV container
// (WAV PCM is U8 for 8-bit and signed for 16/24/32-bit; the unsigned
// 16/24/32-bit and signed-8-bit variants only arise from formats whose
// decoders are not enabled here, e.g. AIFF S8) — see the report.
//
// Oracle discipline: every expected sample is hand-computed from the
// raw bytes written into the WAV body, never produced by calling
// `load_audio`. For an all-silence body each format's "zero codepoint"
// (PCM-U8 midpoint 0x80, signed/float exact-0 byte patterns) decodes to
// exactly 0.0; the exact-value oracles use byte patterns whose decoded
// f32 is a power-of-two multiple (bit-exact under the `/2^(bits-1)` and
// `s as f32` paths).

/// Build a fully-valid 16-byte-fmt RIFF/WAVE file with the given format
/// tag, bit depth, channel count, and sample rate, carrying `body` as
/// the raw little-endian PCM/float data-chunk payload. Independent of
/// the code under test — a hand-assembled header, not a `save_wav`
/// call. `body.len()` must be a whole number of frames
/// (`channels * bits/8`) so the decoded count equals the
/// header-declared `num_frames` (otherwise `load_audio`'s exact-count
/// cross-check would fire — which is the point: a faithful container
/// never trips it).
fn build_wav(format_tag: u16, bits: u16, channels: u16, sample_rate: u32, body: &[u8]) -> Vec<u8> {
  let block_align: u16 = channels * (bits / 8);
  let data_size: u32 = u32::try_from(body.len()).expect("body fits u32");
  let byte_rate: u32 = sample_rate * u32::from(block_align);
  let file_size_minus_8: u32 = 36 + data_size;
  let mut v = Vec::with_capacity(44 + body.len());
  v.extend_from_slice(b"RIFF");
  v.extend_from_slice(&file_size_minus_8.to_le_bytes());
  v.extend_from_slice(b"WAVE");
  v.extend_from_slice(b"fmt ");
  v.extend_from_slice(&16u32.to_le_bytes()); // 16-byte fmt chunk
  v.extend_from_slice(&format_tag.to_le_bytes()); // 1 = PCM, 3 = IEEE float
  v.extend_from_slice(&channels.to_le_bytes());
  v.extend_from_slice(&sample_rate.to_le_bytes());
  v.extend_from_slice(&byte_rate.to_le_bytes());
  v.extend_from_slice(&block_align.to_le_bytes());
  v.extend_from_slice(&bits.to_le_bytes());
  v.extend_from_slice(b"data");
  v.extend_from_slice(&data_size.to_le_bytes());
  v.extend_from_slice(body);
  v
}

/// 8-bit PCM WAV (format tag 1) decodes through the `U8` arm:
/// `out = (i16::from(byte) - 128) / 128.0`. Covers the U8 arm
/// (io.rs:915-925) and `int_divisor(8)` (io.rs:865).
///
/// Oracle (hand-computed from the raw bytes, NOT from `load_audio`):
///   0x80 (128) -> (128-128)/128 =  0.0   (silence / offset-binary midpoint)
///   0xC0 (192) -> (192-128)/128 =  0.5
///   0x40 ( 64) -> ( 64-128)/128 = -0.5
///   0xFF (255) -> (255-128)/128 =  0.9921875
///   0x00 (  0) -> (  0-128)/128 = -1.0
#[test]
fn load_audio_u8_pcm_decodes_through_u8_arm() {
  let dir = audio_temp_dir("audio_u8_pcm");
  let path = dir.join("u8.wav");
  let body: [u8; 5] = [0x80, 0xC0, 0x40, 0xFF, 0x00];
  fs::write(&path, build_wav(1, 8, 1, 8_000, &body)).expect("write u8 WAV");
  let (decoded, sr) = load_audio(&path).expect("8-bit PCM WAV must decode through the U8 arm");
  assert_eq!(sr, 8_000, "sample-rate mismatch");
  let expected = [0.0_f32, 0.5, -0.5, 127.0 / 128.0, -1.0];
  assert_eq!(decoded.len(), expected.len(), "U8 frame-count mismatch");
  for (i, (&got, &exp)) in decoded.iter().zip(expected.iter()).enumerate() {
    assert_eq!(
      got.to_bits(),
      exp.to_bits(),
      "U8 decode mismatch at {i}: got {got}, expected {exp}"
    );
  }
}

/// 24-bit PCM WAV (format tag 1) decodes through the `S24` arm:
/// `out = i24_value * (1.0 / 2^23)`. Covers `int_divisor(24)`
/// (io.rs:867) and the S24 decode arm body.
///
/// Oracle: each 3-byte LE group encodes a signed 24-bit value `v`;
/// `read_i24` sign-extends it and the codec stores `i24(v)` (the
/// `read_i24() << 8` left-shift is undone by the `i24::from(.. >> 8)`
/// conversion). `inner()` yields `v`, normalized by `1/2^23`:
///   [0x00,0x00,0x00] -> v=0          ->  0.0
///   [0x00,0x00,0x40] -> v=0x40_0000  ->  2^22 / 2^23 =  0.5
///   [0x00,0x00,0xC0] -> v=-0x40_0000 -> -2^22 / 2^23 = -0.5
/// (0x40_0000 = 4_194_304 = 2^22; 0xC0_0000 sign-extends to -4_194_304.)
#[test]
fn load_audio_s24_pcm_decodes_through_s24_arm() {
  let dir = audio_temp_dir("audio_s24_pcm");
  let path = dir.join("s24.wav");
  // 3 frames (mono, 3 bytes each).
  let body: [u8; 9] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0xC0];
  fs::write(&path, build_wav(1, 24, 1, 16_000, &body)).expect("write s24 WAV");
  let (decoded, sr) = load_audio(&path).expect("24-bit PCM WAV must decode through the S24 arm");
  assert_eq!(sr, 16_000, "sample-rate mismatch");
  let expected = [0.0_f32, 0.5, -0.5];
  assert_eq!(decoded.len(), expected.len(), "S24 frame-count mismatch");
  for (i, (&got, &exp)) in decoded.iter().zip(expected.iter()).enumerate() {
    assert_eq!(
      got.to_bits(),
      exp.to_bits(),
      "S24 decode mismatch at {i}: got {got}, expected {exp}"
    );
  }
}

/// 32-bit PCM WAV (format tag 1) decodes through the `S32` arm:
/// `out = i32_value * (1.0 / 2^31)`. Covers `int_divisor(32)`
/// (io.rs:868) and the S32 decode arm body.
///
/// Oracle: each 4-byte LE group is a signed 32-bit value `v`,
/// normalized by `1/2^31`:
///   0x0000_0000 -> v=0          ->  0.0
///   0x4000_0000 -> v=2^30       ->  2^30 / 2^31 =  0.5
///   0xC000_0000 -> v=-2^30      -> -2^30 / 2^31 = -0.5
#[test]
fn load_audio_s32_pcm_decodes_through_s32_arm() {
  let dir = audio_temp_dir("audio_s32_pcm");
  let path = dir.join("s32.wav");
  let mut body: Vec<u8> = Vec::new();
  body.extend_from_slice(&0i32.to_le_bytes());
  body.extend_from_slice(&(1i32 << 30).to_le_bytes()); // 2^30
  body.extend_from_slice(&(-(1i32 << 30)).to_le_bytes()); // -2^30
  fs::write(&path, build_wav(1, 32, 1, 44_100, &body)).expect("write s32 WAV");
  let (decoded, sr) = load_audio(&path).expect("32-bit PCM WAV must decode through the S32 arm");
  assert_eq!(sr, 44_100, "sample-rate mismatch");
  let expected = [0.0_f32, 0.5, -0.5];
  assert_eq!(decoded.len(), expected.len(), "S32 frame-count mismatch");
  for (i, (&got, &exp)) in decoded.iter().zip(expected.iter()).enumerate() {
    assert_eq!(
      got.to_bits(),
      exp.to_bits(),
      "S32 decode mismatch at {i}: got {got}, expected {exp}"
    );
  }
}

/// 32-bit IEEE-float WAV (format tag 3) decodes through the `F32` arm
/// (pass-through, no divisor). Covers the F32 decode arm
/// (io.rs:903-908) — which the existing PCM-16 round-trip never
/// exercises — and the `check_finite` Ok path.
///
/// Oracle: the body is raw little-endian IEEE-754 f32, passed through
/// unchanged.
#[test]
fn load_audio_f32_ieee_decodes_through_f32_arm() {
  let dir = audio_temp_dir("audio_f32_ieee");
  let path = dir.join("f32.wav");
  let samples = [0.0_f32, 0.5, -0.5, 0.25, -1.0];
  let mut body: Vec<u8> = Vec::new();
  for &s in &samples {
    body.extend_from_slice(&s.to_le_bytes());
  }
  fs::write(&path, build_wav(3, 32, 1, 48_000, &body)).expect("write f32 IEEE WAV");
  let (decoded, sr) = load_audio(&path).expect("32-bit IEEE-float WAV must decode through F32 arm");
  assert_eq!(sr, 48_000, "sample-rate mismatch");
  assert_eq!(decoded.len(), samples.len(), "F32 frame-count mismatch");
  for (i, (&got, &exp)) in decoded.iter().zip(samples.iter()).enumerate() {
    assert_eq!(
      got.to_bits(),
      exp.to_bits(),
      "F32 pass-through mismatch at {i}: got {got}, expected {exp}"
    );
  }
}

/// 64-bit IEEE-float WAV (format tag 3) decodes through the `F64` arm:
/// `out = (f64_sample as f32)`. Covers the F64 decode arm (io.rs:909-914).
///
/// Oracle: the body is raw little-endian IEEE-754 f64; each value is
/// chosen exactly representable in f32 (powers of two / small dyadic
/// rationals) so the `as f32` narrowing is bit-exact.
#[test]
fn load_audio_f64_ieee_decodes_through_f64_arm() {
  let dir = audio_temp_dir("audio_f64_ieee");
  let path = dir.join("f64.wav");
  let samples_f64 = [0.0_f64, 0.5, -0.5, 0.125, -0.75];
  let mut body: Vec<u8> = Vec::new();
  for &s in &samples_f64 {
    body.extend_from_slice(&s.to_le_bytes());
  }
  fs::write(&path, build_wav(3, 64, 1, 22_050, &body)).expect("write f64 IEEE WAV");
  let (decoded, sr) = load_audio(&path).expect("64-bit IEEE-float WAV must decode through F64 arm");
  assert_eq!(sr, 22_050, "sample-rate mismatch");
  let expected: Vec<f32> = samples_f64.iter().map(|&s| s as f32).collect();
  assert_eq!(decoded.len(), expected.len(), "F64 frame-count mismatch");
  for (i, (&got, &exp)) in decoded.iter().zip(expected.iter()).enumerate() {
    assert_eq!(
      got.to_bits(),
      exp.to_bits(),
      "F64 narrowing mismatch at {i}: got {got}, expected {exp}"
    );
  }
}

/// A WAV whose fmt chunk declares IMA ADPCM (format tag 0x0011) probes
/// successfully (the RIFF reader parses the ADPCM fmt chunk and assigns
/// the ADPCM codec id) but FAILS at `make_audio_decoder` — the `pcm`
/// feature registers only PCM codecs, so there is no ADPCM decoder to
/// build. That surfaces as `Error::Parse` from the
/// `make_audio_decoder` arm (io.rs:526-529).
///
/// The fmt chunk is hand-assembled to satisfy symphonia's
/// `read_adpcm_fmt` requirements (bits_per_sample == 4, fmt len >= 20,
/// IMA `cbSize` extension == 2 with 2 bytes of extra data) so the probe
/// reaches the track-creation stage; only the decoder build fails.
#[test]
fn load_audio_ima_adpcm_wav_fails_at_make_decoder_with_parse() {
  let dir = audio_temp_dir("audio_ima_adpcm");
  let path = dir.join("adpcm.wav");

  // Hand-assemble a 20-byte WAVEFORMATEX fmt chunk for IMA ADPCM:
  //   audioFormat = 0x0011 (WAVE_FORMAT_ADPCM_IMA)
  //   channels = 1, sampleRate = 8000, byteRate, blockAlign,
  //   bitsPerSample = 4, cbSize = 2, wSamplesPerBlock = <2 bytes>.
  const FMT_LEN: u32 = 20;
  let channels: u16 = 1;
  let sample_rate: u32 = 8_000;
  let block_align: u16 = 256; // arbitrary valid ADPCM block size
  let bits: u16 = 4;
  // A tiny data chunk (content is irrelevant — the decoder build fails
  // before any packet is decoded).
  let data: [u8; 4] = [0x00, 0x00, 0x00, 0x00];
  let data_size: u32 = data.len() as u32;
  let byte_rate: u32 = sample_rate; // nominal; symphonia only logs mismatches
  let file_size_minus_8: u32 = 4 /*WAVE*/ + (8 + FMT_LEN) + (8 + data_size);

  let mut v: Vec<u8> = Vec::new();
  v.extend_from_slice(b"RIFF");
  v.extend_from_slice(&file_size_minus_8.to_le_bytes());
  v.extend_from_slice(b"WAVE");
  v.extend_from_slice(b"fmt ");
  v.extend_from_slice(&FMT_LEN.to_le_bytes());
  v.extend_from_slice(&0x0011u16.to_le_bytes()); // WAVE_FORMAT_ADPCM_IMA
  v.extend_from_slice(&channels.to_le_bytes());
  v.extend_from_slice(&sample_rate.to_le_bytes());
  v.extend_from_slice(&byte_rate.to_le_bytes());
  v.extend_from_slice(&block_align.to_le_bytes());
  v.extend_from_slice(&bits.to_le_bytes()); // 16 bytes of WAVEFORMATEX so far
  v.extend_from_slice(&2u16.to_le_bytes()); // cbSize = 2 (IMA requires exactly 2)
  v.extend_from_slice(&505u16.to_le_bytes()); // wSamplesPerBlock (2 bytes of extra)
  v.extend_from_slice(b"data");
  v.extend_from_slice(&data_size.to_le_bytes());
  v.extend_from_slice(&data);

  fs::write(&path, &v).expect("write IMA ADPCM WAV");
  let r = load_audio(&path);
  assert!(
    matches!(r, Err(Error::Parse(_))),
    "an undecodable IMA ADPCM WAV must surface Error::Parse \
       (probe succeeds, make_audio_decoder fails), got {r:?}"
  );
}

/// A non-finite (NaN) sample in an IEEE-float WAV is rejected by the
/// `check_finite` guard with `Error::NonFiniteScalar` — a NaN/inf in the
/// decoded PCM would silently poison downstream DSP. Covers the
/// `check_finite` Err arm (io.rs:852-854) through the F32 decode path.
///
/// The NaN is hand-encoded as the raw f32 bit pattern in the WAV body
/// (`f32::NAN.to_le_bytes()`), so the rejection is driven by real
/// decoded bytes, not a synthetic in-memory sample.
#[test]
fn load_audio_non_finite_float_sample_is_non_finite_scalar() {
  let dir = audio_temp_dir("audio_f32_nan");
  let path = dir.join("nan.wav");
  // First sample finite, second a NaN — so the guard fires mid-buffer.
  let mut body: Vec<u8> = Vec::new();
  body.extend_from_slice(&0.25_f32.to_le_bytes());
  body.extend_from_slice(&f32::NAN.to_le_bytes());
  fs::write(&path, build_wav(3, 32, 1, 16_000, &body)).expect("write NaN-bearing f32 WAV");
  let r = load_audio(&path);
  assert!(
    matches!(r, Err(Error::NonFiniteScalar(_))),
    "a NaN PCM sample must be rejected with NonFiniteScalar, got {r:?}"
  );
}

/// An infinity (+inf) f32 sample is likewise rejected by `check_finite`
/// (io.rs:852-854) — `is_finite()` covers both NaN and ±inf.
#[test]
fn load_audio_infinite_float_sample_is_non_finite_scalar() {
  let dir = audio_temp_dir("audio_f32_inf");
  let path = dir.join("inf.wav");
  let mut body: Vec<u8> = Vec::new();
  body.extend_from_slice(&f32::INFINITY.to_le_bytes());
  fs::write(&path, build_wav(3, 32, 1, 16_000, &body)).expect("write +inf f32 WAV");
  let r = load_audio(&path);
  assert!(
    matches!(r, Err(Error::NonFiniteScalar(_))),
    "a +inf PCM sample must be rejected with NonFiniteScalar, got {r:?}"
  );
}

// ---- CapStrategy::resolve: duration-cap arithmetic + saturation -------

/// `CapStrategy::SrcRateMaxSeconds::resolve` clamps the
/// `src_sr * max_seconds` product to `MAX_DECODED_SAMPLES` and, on a
/// non-finite product (the defensive guard for a future caller that
/// bypasses the public `load_audio_with_max_seconds` validation),
/// falls back to `MAX_DECODED_SAMPLES`. Covers the non-finite
/// fallback arm (io.rs:744) plus the finite/clamping arms.
///
/// Called directly on the private enum (in-module test) so the
/// validated public entry point doesn't shadow the defensive branch.
/// Oracle is hand-computed: the products are exact integers in f64.
#[test]
fn cap_strategy_resolve_src_rate_max_seconds_arithmetic_and_saturation() {
  // Finite, well under the cap: 16_000 Hz * 2.0 s = 32_000 samples.
  assert_eq!(
    CapStrategy::SrcRateMaxSeconds(2.0).resolve(16_000),
    32_000,
    "src_sr * max_seconds must be the in-cap product"
  );

  // Finite but the product exceeds the global ceiling -> clamp to
  // MAX_DECODED_SAMPLES. 1_000_000 Hz * 1_000_000 s = 1e12 >> 64 Mi.
  assert_eq!(
    CapStrategy::SrcRateMaxSeconds(1_000_000.0).resolve(1_000_000),
    MAX_DECODED_SAMPLES,
    "an over-ceiling product must clamp to MAX_DECODED_SAMPLES"
  );

  // Non-finite product (NaN) -> the defensive else-branch returns
  // MAX_DECODED_SAMPLES (io.rs:744). The public entry validates
  // max_seconds first, so this branch is only reachable by calling
  // resolve directly.
  assert_eq!(
    CapStrategy::SrcRateMaxSeconds(f32::NAN).resolve(16_000),
    MAX_DECODED_SAMPLES,
    "a NaN product must fall back to MAX_DECODED_SAMPLES, not panic/wrap"
  );

  // +inf product -> same defensive fallback (is_finite() is false).
  assert_eq!(
    CapStrategy::SrcRateMaxSeconds(f32::INFINITY).resolve(16_000),
    MAX_DECODED_SAMPLES,
    "an +inf product must fall back to MAX_DECODED_SAMPLES"
  );

  // The explicit-cap arm clamps to the global ceiling too (the
  // MaxSamples branch of resolve).
  assert_eq!(
    CapStrategy::MaxSamples(usize::MAX).resolve(16_000),
    MAX_DECODED_SAMPLES,
    "MaxSamples(usize::MAX) must clamp to the global ceiling"
  );
  assert_eq!(
    CapStrategy::MaxSamples(1234).resolve(16_000),
    1234,
    "an in-ceiling MaxSamples must pass through unchanged"
  );
}

/// `open_excl_tempfile` against a destination whose PARENT directory
/// does not exist makes `OpenOptions::create_new` fail with a
/// non-`AlreadyExists` error (`NotFound`), exercising the generic
/// create-failure arm that returns `Error::FileIo` with
/// `FileOp::Create` (io.rs:1937-1944) — distinct from the
/// retry-exhaustion arm (which the zero-retry test already covers).
#[test]
fn open_excl_tempfile_nonexistent_parent_is_fileio_create_error() {
  // A parent dir that does not exist on disk -> create_new fails with
  // NotFound, NOT AlreadyExists, so the generic `Err(e)` arm fires on
  // the very first retry.
  let missing_parent = Path::new("/mlxrs_no_such_parent_dir_for_tempfile_test_zzz/dest.wav");
  let r = open_excl_tempfile(missing_parent, 4);
  match r {
    Err(Error::FileIo(p)) => {
      assert_eq!(
        p.op(),
        FileOp::Create,
        "a non-AlreadyExists create failure must carry FileOp::Create"
      );
    }
    other => panic!("expected FileIo(Create) for a missing parent dir, got {other:?}"),
  }
}

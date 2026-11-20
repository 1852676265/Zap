//! Team-local session binding: never let a foreign session restore its project root.
use std::path::{Path, PathBuf};

use zap_capabilities::session::SessionManager as NativeSessionManager;

#[derive(Debug)]
enum Binding {
    Absent,
    Bound(PathBuf),
    // A display/snapshot entry without readable native metadata is not binding proof.
    Unknown,
}

pub(crate) fn same_project(a: &Path, b: &Path) -> bool {
    NativeSessionManager::project_hash(a) == NativeSessionManager::project_hash(b)
}

pub(crate) fn session_id_for_binding(
    existing_id: &str,
    base_id: &str,
    working_dir: &Path,
) -> Result<String, String> {
    session_id_with_resolver(existing_id, base_id, working_dir, |id| {
        resolve_binding_in_root(&NativeSessionManager::sessions_root(), id)
    })
}

fn resolve_binding_in_root(root: &Path, id: &str) -> Result<Binding, String> {
    // Validate the exact ID using the native path rules, not business-ID shapes.
    NativeSessionManager::with_root(root)
        .meta_path(id)
        .map_err(|e| e.to_string())?;
    let buckets = match std::fs::read_dir(root) {
        Ok(buckets) => buckets,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Binding::Absent),
        Err(error) => return Err(error.to_string()),
    };
    // Catalog lookup omits display-only entries and can select just one of several
    // buckets with the same ID. Neither behavior is sufficient binding proof.
    let mut binding = Binding::Absent;
    for bucket in buckets {
        let bucket = bucket.map_err(|e| e.to_string())?;
        let file_type = bucket.file_type().map_err(|e| e.to_string())?;
        if file_type.is_symlink() {
            // We cannot prove what a redirected bucket might restore.
            return Err("会话 project bucket 是符号链接，无法安全确认绑定".into());
        }
        if !file_type.is_dir() {
            continue;
        }
        let manager = NativeSessionManager::with_root(bucket.path());
        // 检查原生存储公开的权威/展示文件路径，避免另维护一份文件后缀注册表。
        let artifacts = [
            manager.meta_path(id),
            manager.snapshot_path(id),
            manager.legacy_path(id),
            manager.jsonl_path(id),
            manager.presentation_path(id),
            manager.display_path(id),
            manager.display_meta_path(id),
            manager.display_index_path(id),
        ];
        let mut present = false;
        for path in artifacts {
            match std::fs::symlink_metadata(path.map_err(|e| e.to_string())?) {
                Ok(_) => present = true,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.to_string()),
            }
        }
        if !present {
            continue;
        }
        // Even matching metadata in one bucket must not hide another bucket:
        // process restore may choose a different copy. Isolate any duplicate ID.
        if !matches!(binding, Binding::Absent) {
            return Ok(Binding::Unknown);
        }
        binding = match manager.read_meta(id) {
            Ok(meta) => Binding::Bound(PathBuf::from(meta.working_dir)),
            Err(_) => Binding::Unknown,
        };
    }
    Ok(binding)
}

fn session_id_with_resolver(
    existing_id: &str,
    base_id: &str,
    working_dir: &Path,
    mut resolve: impl FnMut(&str) -> Result<Binding, String>,
) -> Result<String, String> {
    if base_id.trim().is_empty() || working_dir.as_os_str().is_empty() {
        return Err("会话绑定需要 base ID 和工作目录".into());
    }
    for id in [existing_id, base_id] {
        if !id.is_empty() {
            if let Binding::Bound(path) = resolve(id)? {
                if same_project(&path, working_dir) {
                    return Ok(id.to_string());
                }
            }
        }
    }
    let isolated = format!(
        "{base_id}-{}",
        NativeSessionManager::project_hash(working_dir)
    );
    match resolve(&isolated)? {
        Binding::Absent => return Ok(isolated),
        Binding::Bound(path) if same_project(&path, working_dir) => return Ok(isolated),
        _ => {}
    }
    // Do not overwrite, move or delete an incorrectly bound deterministic ID.
    loop {
        let fresh = format!("{isolated}-{}", uuid::Uuid::new_v4());
        if matches!(resolve(&fresh)?, Binding::Absent) {
            return Ok(fresh);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_only_candidate_requires_uuid_without_touching_old_files() {
        let root = tempfile::tempdir().unwrap();
        let wd = root.path().join("current");
        let base = "business.display-id";
        let candidate = format!("{base}-{}", NativeSessionManager::project_hash(&wd));
        for suffix in [
            "display.jsonl",
            "display.meta.json",
            "display.idx.jsonl",
            "snapshot",
        ] {
            let sessions = tempfile::tempdir().unwrap();
            let bucket = sessions.path().join("0123456789abcdef");
            std::fs::create_dir(&bucket).unwrap();
            let artifact = bucket.join(format!("{candidate}.{suffix}"));
            std::fs::write(&artifact, b"old session artifact").unwrap();
            assert!(matches!(
                resolve_binding_in_root(sessions.path(), &candidate).unwrap(),
                Binding::Unknown
            ));
            let fresh = session_id_with_resolver("", base, &wd, |id| {
                resolve_binding_in_root(sessions.path(), id)
            })
            .unwrap();
            let uuid = fresh.strip_prefix(&format!("{candidate}-")).unwrap();
            assert!(uuid::Uuid::parse_str(uuid).is_ok());
            assert_eq!(std::fs::read(&artifact).unwrap(), b"old session artifact");
        }
    }

    #[test]
    fn exact_artifacts_do_not_match_neighbor_ids() {
        let root = tempfile::tempdir().unwrap();
        let bucket = root.path().join("0123456789abcdef");
        std::fs::create_dir(&bucket).unwrap();
        for name in ["prefix-business.display.jsonl", "business-extra.snapshot"] {
            std::fs::write(bucket.join(name), b"old").unwrap();
        }
        assert!(matches!(
            resolve_binding_in_root(root.path(), "business").unwrap(),
            Binding::Absent
        ));
    }

    #[test]
    fn duplicate_id_never_reuses_only_the_newest_same_project_meta() {
        use zap_capabilities::session::SessionMeta;

        for duplicate in ["foreign", "same", "display-only", "unreadable"] {
            let root = tempfile::tempdir().unwrap();
            let wd = root.path().join("current");
            let base = "business";
            let candidate = format!("{base}-{}", NativeSessionManager::project_hash(&wd));
            let first = root.path().join("0123456789abcdef");
            let second = root.path().join("fedcba9876543210");
            std::fs::create_dir(&first).unwrap();
            std::fs::create_dir(&second).unwrap();
            let manager = NativeSessionManager::with_root(&first);
            let meta = SessionMeta::new(&candidate, wd.to_string_lossy().into_owned(), 200);
            manager.write_meta(&meta).unwrap();
            assert!(matches!(
                resolve_binding_in_root(root.path(), &candidate).unwrap(),
                Binding::Bound(_)
            ));
            let other = NativeSessionManager::with_root(&second);
            match duplicate {
                "foreign" | "same" => {
                    let other_wd = if duplicate == "same" {
                        wd.clone()
                    } else {
                        root.path().join("old")
                    };
                    other
                        .write_meta(&SessionMeta::new(
                            &candidate,
                            other_wd.to_string_lossy().into_owned(),
                            100,
                        ))
                        .unwrap();
                }
                "display-only" => {
                    std::fs::write(second.join(format!("{candidate}.display.jsonl")), b"old")
                        .unwrap()
                }
                _ => std::fs::write(other.meta_path(&candidate).unwrap(), b"broken").unwrap(),
            }
            assert!(matches!(
                resolve_binding_in_root(root.path(), &candidate).unwrap(),
                Binding::Unknown
            ));
            let fresh = session_id_with_resolver(&candidate, base, &wd, |id| {
                resolve_binding_in_root(root.path(), id)
            })
            .unwrap();
            assert!(
                uuid::Uuid::parse_str(fresh.strip_prefix(&format!("{candidate}-")).unwrap())
                    .is_ok()
            );
            assert!(manager.read_meta(&candidate).is_ok());
        }
    }

    #[test]
    fn binding_requires_metadata_not_id_shape_or_display() {
        let wd = Path::new("/team/current");
        let candidate = format!("sql-fixed-{}", NativeSessionManager::project_hash(wd));
        let choose = |existing: &str| {
            session_id_with_resolver(existing, "sql-fixed", wd, |id| {
                Ok(if id == existing {
                    Binding::Bound("/team/old".into())
                } else {
                    Binding::Absent
                })
            })
            .unwrap()
        };
        assert_eq!(choose("member-old"), candidate);
        assert_eq!(choose("sql-fixed-fake-project-hash"), candidate);
    }

    #[test]
    fn stable_isolation_and_poisoned_candidate_fallback() {
        let wd = Path::new("/team/current");
        let isolated = session_id_with_resolver("", "base", wd, |_| Ok(Binding::Absent)).unwrap();
        assert_eq!(
            isolated,
            session_id_with_resolver("", "base", wd, |_| Ok(Binding::Absent)).unwrap()
        );
        for poisoned in [false, true] {
            let fresh = session_id_with_resolver("old", "base", wd, |id| {
                Ok(if id == isolated {
                    if poisoned {
                        Binding::Bound("/other".into())
                    } else {
                        Binding::Unknown
                    }
                } else {
                    Binding::Absent
                })
            })
            .unwrap();
            assert_ne!(fresh, isolated);
        }
        for reusable in ["old", "base"] {
            assert_eq!(
                session_id_with_resolver("old", "base", wd, |id| {
                    Ok(if id == reusable {
                        Binding::Bound("/team/current/".into())
                    } else {
                        Binding::Unknown
                    })
                })
                .unwrap(),
                reusable
            );
        }
        assert!(session_id_with_resolver("", "base", wd, |_| Err("io".into())).is_err());
    }
}

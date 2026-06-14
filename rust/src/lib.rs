//! FraudGuard Rust core (pyo3).
//!
//! CPU-bound hot paths ported from Python for speed, with identical results so
//! the pure-Python fallbacks in `src/` stay drop-in compatible:
//!   - `detect_rings`  : fraud-ring detection via union-find on the user/infra
//!                       bipartite graph (the graph layer's heavy loop)
//!   - `velocity_24h`  : per-user count of prior transactions within 24h
//!                       (the transaction feature kernel run right after the simulator)

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;

// ----------------------------------------------------------------- union-find
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind { parent: (0..n).collect(), rank: vec![0; n] }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // path compression
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// Detect fraud rings from user<->infrastructure edges.
///
/// `edges` is a list of (user_node, infra_node) pairs (devices/IPs/identities a
/// user touches). Returns one tuple per ring of at least `min_size` users:
/// (members, size, n_shared_identifiers, density, risk_score).
#[pyfunction]
fn detect_rings(
    edges: Vec<(String, String)>,
    min_size: usize,
) -> Vec<(Vec<String>, usize, usize, f64, f64)> {
    // intern users
    let mut user_idx: HashMap<String, usize> = HashMap::new();
    let mut user_names: Vec<String> = Vec::new();
    let mut infra_users: HashMap<String, Vec<usize>> = HashMap::new();

    for (u, infra) in &edges {
        let idx = match user_idx.get(u) {
            Some(&i) => i,
            None => {
                let i = user_names.len();
                user_idx.insert(u.clone(), i);
                user_names.push(u.clone());
                i
            }
        };
        infra_users.entry(infra.clone()).or_default().push(idx);
    }

    let mut uf = UnionFind::new(user_names.len());
    for users in infra_users.values() {
        if users.len() >= 2 {
            let first = users[0];
            for &u in &users[1..] {
                uf.union(first, u);
            }
        }
    }

    // group users by component
    let mut comp_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..user_names.len() {
        let r = uf.find(i);
        comp_members.entry(r).or_default().push(i);
    }

    // shared infra (>=2 users) attributed to their component root, with user lists
    let mut comp_infra: HashMap<usize, Vec<Vec<usize>>> = HashMap::new();
    for users in infra_users.values() {
        if users.len() >= 2 {
            let mut deduped = users.clone();
            deduped.sort_unstable();
            deduped.dedup();
            if deduped.len() >= 2 {
                let root = uf.find(deduped[0]);
                comp_infra.entry(root).or_default().push(deduped);
            }
        }
    }

    let mut rings = Vec::new();
    for (root, members) in &comp_members {
        let n = members.len();
        if n < min_size {
            continue;
        }
        let infra_lists = comp_infra.get(root);
        let n_shared = infra_lists.map(|v| v.len()).unwrap_or(0);

        // user-pair edges from each shared-infra clique
        let mut pairs: HashSet<(usize, usize)> = HashSet::new();
        if let Some(lists) = infra_lists {
            for users in lists {
                for i in 0..users.len() {
                    for j in (i + 1)..users.len() {
                        let (a, b) = (users[i], users[j]);
                        pairs.insert(if a < b { (a, b) } else { (b, a) });
                    }
                }
            }
        }
        let e = pairs.len() as f64;
        let density = if n >= 2 {
            2.0 * e / (n as f64 * (n as f64 - 1.0))
        } else {
            0.0
        };
        let size_term = (n as f64 / 8.0).min(1.0);
        let shared_term = (n_shared as f64 / 4.0).min(1.0);
        let risk = 0.4 * size_term + 0.35 * density + 0.25 * shared_term;

        let mut names: Vec<String> = members.iter().map(|&i| user_names[i].clone()).collect();
        names.sort();
        rings.push((names, n, n_shared, density, risk));
    }
    rings
}

/// For each transaction (parallel arrays), count the same user's prior
/// transactions within the preceding 24h. Mirrors the Python implementation:
/// per user, on timestamps sorted ascending, count = position - first index
/// with ts >= current_ts - 86400.
#[pyfunction]
fn velocity_24h(user_ids: Vec<String>, ts: Vec<i64>) -> Vec<i64> {
    let n = user_ids.len();
    let mut result = vec![0i64; n];
    let mut by_user: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, u) in user_ids.iter().enumerate() {
        by_user.entry(u.as_str()).or_default().push(i);
    }
    for idxs in by_user.values() {
        // sort original indices by timestamp ascending (stable)
        let mut order: Vec<usize> = idxs.clone();
        order.sort_by_key(|&i| ts[i]);
        let times: Vec<i64> = order.iter().map(|&i| ts[i]).collect();
        for p in 0..times.len() {
            let threshold = times[p] - 86_400;
            // first index where times[lo] >= threshold
            let lo = times.partition_point(|&t| t < threshold);
            result[order[p]] = (p - lo) as i64;
        }
    }
    result
}

#[pymodule]
fn fraudguard_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_rings, m)?)?;
    m.add_function(wrap_pyfunction!(velocity_24h, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}

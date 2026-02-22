from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass
class GraphRAGRetriever:
    k_hop: int = 2

    def retrieve(self, graph: dict[str, Any], query_terms: list[str], top_k: int = 8) -> dict[str, Any]:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        idx = {n["id"]: n for n in nodes}
        adj: dict[str, set[str]] = defaultdict(set)
        for e in edges:
            adj[e["source"]].add(e["target"])
            adj[e["target"]].add(e["source"])
        seed_nodes = [n["id"] for n in nodes if self._match(n, query_terms)]
        visited = set(seed_nodes)
        q = deque([(n, 0) for n in seed_nodes])
        while q:
            current, depth = q.popleft()
            if depth >= self.k_hop:
                continue
            for nxt in adj[current]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, depth + 1))
        selected_nodes = [idx[nid] for nid in list(visited)[:top_k]]
        selected_ids = {n["id"] for n in selected_nodes}
        selected_edges = [e for e in edges if e["source"] in selected_ids and e["target"] in selected_ids]
        communities = self._components(selected_nodes, selected_edges)
        return {
            "nodes": selected_nodes,
            "edges": selected_edges,
            "seed_nodes": seed_nodes,
            "stats": {"retrieved_nodes": len(selected_nodes), "retrieved_edges": len(selected_edges)},
            "community_summary": communities,
        }

    def _match(self, node: dict[str, Any], terms: list[str]) -> bool:
        text = f"{node.get('label', '')} {' '.join(node.get('synonyms', []))}".lower()
        return any(t.lower() in text for t in terms)

    def _components(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ids = [n["id"] for n in nodes]
        adj = defaultdict(set)
        for e in edges:
            adj[e["source"]].add(e["target"])
            adj[e["target"]].add(e["source"])
        seen = set()
        comps: list[dict[str, Any]] = []
        for nid in ids:
            if nid in seen:
                continue
            stack = [nid]
            comp = []
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                comp.append(cur)
                stack.extend(adj[cur] - seen)
            comps.append({"size": len(comp), "members": comp[:5]})
        return sorted(comps, key=lambda x: x["size"], reverse=True)

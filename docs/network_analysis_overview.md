# Network Analysis Overview

## Network Construction

The code constructs an undirected weighted author collaboration network from the
cleaned EPB publication CSV.

- Nodes are standardized author names from `authors_full_final`.
- Edges represent co-authorship on at least one paper.
- Single-authored papers add an author node but do not create an edge.

For each multi-authored paper `p` with `n_p` unique authors, every co-author pair
on that paper receives a paper-level contribution:

```text
w_ij^(p) = 1 / n_p
```

If the same two authors co-authored multiple papers, their edge weight is:

```text
W_ij = sum over papers p containing both i and j of (1 / n_p)
```

## Main Measures

- `paper_count`: number of EPB papers in which an author appears.
- `unweighted degree`: number of distinct collaborators.
- `weighted degree`: sum of all collaboration edge weights incident to an author.
- `average path length` and `diameter`: computed on the LCC.
- `community detection`: Louvain algorithm on the LCC with `weight="weight"` and
  random seed `42`.

## Network Scope

The code reports both:

- the full `author collaboration network`
- the `largest connected component (LCC) of the author collaboration network`

Degree-based author measures are calculated on the full network. Path-based
measures and Louvain community detection are calculated on the LCC.

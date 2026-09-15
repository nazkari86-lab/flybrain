# MaleCNS v1.0 Essential Local Dataset

Verified on 2026-09-16 from the official HHMI Janelia download endpoints listed at
<https://male-cns.janelia.org/download/>.

| Product | Bytes | Arrow rows | SHA-256 |
|---|---:|---:|---|
| body annotations | 14,483,314 | 211,577 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| body neurotransmitters | 43,282,834 | 1,835,518 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| body stats | 778,062,826 | 88,384,522 | `ca5dc83a26382ae70c8d8f42fc09ce2dbc1af7c03f3a001a1936b5e142540647` |
| connectome weights | 1,051,241,946 | 151,856,684 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |

Total local payload: 1,887,070,920 bytes. Files are stored under the ignored
`data/cache/sha256/` content-addressed tree; the committed declaration is
`data/manifests/male-cns-v1.0-essential.json`.

The annotation row count is not a neuron-count claim. Source tables use different units and
filters. The streaming adapter now applies the explicit valid-superclass and minimum-weight policy;
its measured graph counts are documented in `male-cns-v1.0-import.md`.

The following official products were deliberately deferred because their combined size would
violate the machine's 10 GiB free-space reserve:

- synapse points: 13,061,489,098 bytes;
- synapse partners: 6,777,179,098 bytes;
- T-bar neurotransmitter predictions: 2,651,680,218 bytes.

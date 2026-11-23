# Luma graph schema

Two ideas: **layers** (what is locked vs. extensible) and **perception expires, memory doesn't**.

| Layer | Node labels | Lifetime | Agent can create? |
|---|---|---|---|
| Media | `Session`, `Image`, `AudioClip`, `TextBlock` | days (TTL) | no |
| Perception | `Detection`, `Utterance` | hours (TTL) | no |
| World: things | `:Entity` + `Person`, `Object`, `Place`, `Document`, `Medication` | permanent | yes, via the registry |
| World: events | `:Event` + `Sighting`, `Memory`, `Appointment`, `Reminder` | permanent | yes, via the registry |
| Registry | `EntityType` | permanent | `define_type` only |

Persistent *things* carry the base label `:Entity`; durable *events* carry `:Event`. Generic
queries use the base label; specific ones use the second label (`:Entity:Object`).

## Flow
```
Image -CONTAINS-> Detection -SIGHTING_OF-> Entity            (working memory, pruned)
Detection -PROMOTED_TO-> Sighting -OF-> Entity, -IN-> Place  (long-term, kept)
Utterance/TextBlock -MENTIONS-> Entity ; Memory -ABOUT-> Entity
Appointment -WITH-> Person, -AT-> Place ; Reminder -ABOUT-> Appointment|Medication|...
Person -KNOWS {relationship}-> Person ; Object -OWNED_BY-> Person
Document -HAS_TEXT-> TextBlock ; Document -DESCRIBES-> Medication
```

## Conventions
- Every node has a string `id`. Timestamps are datetimes (`at`, `captured_at`, `created_at`, `start`, `due`).
- Embeddings live in `embedding` (list of floats) on `Detection`, `Image`, `TextBlock`, `Entity`.
- Binary media stays on disk; the graph stores only `uri`.
- Agent-defined types go through `define_type()`: names are normalised (PascalCase labels,
  UPPER_SNAKE relationships), exact matches are reused, near-duplicates are flagged.
- Agent-authored Cypher runs on a read-only connection; writes only go through tools.
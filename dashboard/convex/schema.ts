import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

// Each pushed artifact (ledger CSV, status JSON, picks, …) is stored as raw
// text keyed by its canonical key. The dashboard reads them back over HTTP.
export default defineSchema({
  files: defineTable({
    key: v.string(),
    content: v.string(),
    updatedAt: v.number(),
  }).index("by_key", ["key"]),
});

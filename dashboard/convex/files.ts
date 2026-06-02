import { internalMutation, query } from "./_generated/server";
import { v } from "convex/values";

export const set = internalMutation({
  args: { key: v.string(), content: v.string() },
  handler: async (ctx, { key, content }) => {
    const existing = await ctx.db
      .query("files")
      .withIndex("by_key", (q) => q.eq("key", key))
      .unique();
    if (existing) await ctx.db.patch(existing._id, { content, updatedAt: Date.now() });
    else await ctx.db.insert("files", { key, content, updatedAt: Date.now() });
  },
});

export const get = query({
  args: { key: v.string() },
  handler: async (ctx, { key }) => {
    const row = await ctx.db
      .query("files")
      .withIndex("by_key", (q) => q.eq("key", key))
      .unique();
    return row?.content ?? null;
  },
});

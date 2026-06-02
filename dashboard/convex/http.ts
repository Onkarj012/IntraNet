import { httpRouter } from "convex/server";
import { httpAction } from "./_generated/server";
import { internal, api } from "./_generated/api";

const http = httpRouter();

// Pipeline → Convex. Auth via the PUSH_SECRET Convex env var.
http.route({
  path: "/push",
  method: "POST",
  handler: httpAction(async (ctx, request) => {
    const secret = process.env.PUSH_SECRET;
    if (!secret || request.headers.get("Authorization") !== `Bearer ${secret}`)
      return new Response("unauthorized", { status: 401 });
    let body: { key?: unknown; content?: unknown };
    try {
      body = await request.json();
    } catch {
      return new Response("bad json", { status: 400 });
    }
    if (typeof body.key !== "string" || typeof body.content !== "string")
      return new Response("key and content required", { status: 400 });
    await ctx.runMutation(internal.files.set, { key: body.key, content: body.content });
    return new Response("ok");
  }),
});

// Dashboard ← Convex (server-to-server read; same PUSH_SECRET as /push).
http.route({
  path: "/file",
  method: "GET",
  handler: httpAction(async (ctx, request) => {
    const secret = process.env.PUSH_SECRET;
    if (!secret || request.headers.get("Authorization") !== `Bearer ${secret}`)
      return new Response("unauthorized", { status: 401 });
    const key = new URL(request.url).searchParams.get("key");
    if (!key) return new Response("", { status: 400 });
    const content = await ctx.runQuery(api.files.get, { key });
    return new Response(content ?? "", {
      headers: { "content-type": "text/plain; charset=utf-8" },
    });
  }),
});

export default http;

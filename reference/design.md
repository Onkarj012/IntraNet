# Ovalux — Design Guide

**Type:** Finance dashboard / fintech analytics product landing page
**Personality:** Dark, data-dense, precise, trustworthy. A near-black UI lit by an **indigo-violet** brand accent, with classic **finance semantic colors** (green up / red down / amber warning). Lots of small, tight, medium-weight text — it reads like a real, polished analytics product, not a flashy marketing page. Animated diagonal "data motion" lines add quiet energy.

---

## 1. Aesthetic & Art Direction

- **Mood:** Professional, analytical, premium-fintech. Dark dashboard aesthetic with crisp data widgets, charts, and KPI cards.
- **Color discipline:** Monochrome dark base + a single **indigo-violet** brand accent (`#6e68ee`) + a small set of **semantic data colors** (green/red/amber) reserved for numbers and status.
- **Density:** Information-dense. Base text is small (**14px**) and **medium weight (500)** — the feel of a working product UI.
- **Signature motion:** Animated **moving diagonal hatch lines** (`moveDiagonalLines`) running through panels/backgrounds — a subtle "live data" texture.
- **Single dark theme.**

---

## 2. Color Palette

### Backgrounds & surfaces (deepest → lighter)
| Role | Hex |
|------|-----|
| Page background | `#030303` (near-pure black) |
| Base surface | `#080808` / `#0b0b0b` |
| Panel | `#0f0f0f` / `#121212` |
| Raised card | `#171717` |
| Mid surface | `#2e2e2e` |

### Text
| Role | Hex |
|------|-----|
| Primary | `#ffffff` |
| High-emphasis | `rgba(255,255,255,0.85)` (`#ffffffd9`) |
| Secondary | `rgba(255,255,255,0.6)` (`#fff9`) |
| Muted | `#9c9c9c` |
| Borders | `rgba(255,255,255,0.15)` (`#ffffff26`), `rgba(255,255,255,0.2)` (`#fff3`) |

### Brand accent (violet family)
| Role | Hex |
|------|-----|
| **Primary accent** | `#6e68ee` (indigo-violet) |
| Purple shade | `#744fb8` |
| Muted purple | `#5d5aa3` |
| Blue-violet | `#588bb8` |

### Semantic / data colors (finance)
| Meaning | Hex |
|---------|-----|
| Positive / up | `#22cb58` (green) |
| Negative / down | `#ff5e5d` (red) |
| Warning / neutral-hot | `#ffbc4f` (amber) |

**Usage rule:** Indigo `#6e68ee` is the brand/CTA color. Green/red/amber appear *only* on data — deltas, trends, chart series, status pills. Keep everything else grayscale.

---

## 3. Typography

### Font family
- **Inter** (sans-serif) — the **only** typeface. Used for headings, UI, data, and labels. Clean, neutral, screen-optimized — ideal for dense numeric dashboards.

### Weights
- `500` medium — **the default** (dominant), gives UI a confident, solid feel
- `400` regular — long-form body
- `600` semibold — emphasized labels/subheads
- `700` / `900` — big stat numbers and strong headlines

### Type scale (px)
| Use | Size |
|-----|------|
| Hero | 48px |
| Section title | 36px |
| Sub-section | 22px |
| Lead | 18px |
| Body / UI | 16px |
| **Base UI / data** | **14px** (dominant — dashboard text) |
| Micro / labels | 12px |

### Letter-spacing
- Body/UI: `0px`
- Uppercase labels: `1.8px` (tracked-out caps for eyebrows/section tags)
- Headings: `-0.02em`

### Line-height
- **Data / tight UI: `1`** (dominant — numbers and compact rows sit tight)
- Headings: `120%`
- Specific px line-heights for precise widgets (55/43/31/27/26/20/18px)

### Behaviors
- Small, medium-weight, tight-leading text everywhere — the discipline of a real dashboard.
- Uppercase tracked labels (`1.8px`) for section eyebrows.
- Big bold/black stat numbers as focal points.

---

## 4. Layout & Spacing

- **Breakpoint:** mobile-first; primary desktop layout at `≥810px`.
- **Container:** centered marketing sections; dashboard mockups span wide.
- **Composition:** KPI cards, chart panels, and data tables arranged in tight grids; generous section padding around dense widget clusters.
- **Pattern:** alternating marketing copy + realistic dashboard UI screenshots/mockups.

---

## 5. Shape Language & Radii

| Radius | Use | Frequency |
|--------|-----|-----------|
| `4px` | Small data chips, table cells, tags | most (74×) |
| `8px` | Inputs, small cards | very common |
| `12px` | **Standard card** | common (59×) |
| `16px` | Larger cards | secondary |
| `32px` | Big containers / hero panels | feature (50×) |
| `500px` / `100%` | Pills, circular avatars/icons | accents |
| `31px` | Specific large panels | rare |

**Signature:** layered radii — tiny `4px` data elements nested inside `12–16px` cards inside `32px` section containers. Reflects a real component hierarchy.

---

## 6. Elevation & Shadows

- **Soft multi-layer:** `rgba(0,0,0,0.02) 0 0 0 1px, rgba(0,0,0,0.02) 0 1px 1px 0.5px, ...` — very subtle stacked shadows for floating widgets.
- Realistic float: `rgba(0,0,0,0.17) 0 0.6px 1.6px -1.5px, rgba(0,0,0,0.14) 0 2.3px 6px ...`.
- **Inset hairline:** `inset 0 0 0 1px rgb(0,0,0)` to crisp-edge dark cards.
- Shadows are restrained; depth comes from surface-color steps + hairline borders.

---

## 7. Gradients & Effects

- **Edge/vignette masks (heavy use):** many linear + radial black gradients fade charts and panels into the background — e.g. `linear-gradient(0deg, rgba(0,0,0,0) 0%, rgb(0,0,0) 52%)`, `radial-gradient(50% 50%, rgb(0,0,0) 0%, rgba(0,0,0,0) 100%)`, and `linear-gradient(to right, transparent 0%, #000 12.5%, #000 87.5%, transparent 100%)` to feather marquee/table edges.
- **Highlight sweep:** `linear-gradient(90deg, #fff 0%, rgba(255,255,255,0.6) 100%)` for shiny edges/headlines.
- **Blur glass:** `filter: blur(5px)` (and `10px`) on glowing/soft background elements.
- Indigo glow behind hero and key charts.

---

## 8. Animation & Motion

**Dominant transition:** `transition: color 0.2s ease` (everywhere) + `background-color 0.2s ease, border-color 0.2s ease` on interactive elements. Brand easing for special moves: `cubic-bezier(0.25, 0.75, 0.25, 1)` (smooth, slightly eager ease-out); occasionally `cubic-bezier(0.7, 0, 0.3, 1)`.

### Signature looping animations
- **`moveDiagonalLines-right` — `0.7s linear infinite`** (used 8×): continuously scrolling diagonal hatch lines — a "data flow / activity" texture across panels.
- **`moveDiagonalLines-top-left` — `1.6s linear infinite`**: slower diagonal drift in another direction.
- **`cursor-blink` — `0.8s steps(1) infinite`**: blinking caret in input/terminal mocks.

### Chart / data motion
- **Bar/column growth:** `transform: translate3d(0, 8.5px, 0) scale(1, 0.89)` → `translate3d(0,32.7px,0) scale(1, 0.58)` — charts animate their bars growing/scaling from the baseline.
- Smooth `translate3d` parallax on chart layers and floating tokens.

### Entrance (Framer "appear")
- Elements fade in from near-zero opacity (`opacity: 0.0018`) and settle — gentle fade-up reveals on scroll.

### Performance
- Heavy `will-change: transform` and `translate3d`/`translateZ(0)` to keep chart animations GPU-smooth.

### Motion principles to replicate
1. **Animated diagonal hatch lines** drifting through panels (the signature texture).
2. Charts that **grow/scale their bars** on entry.
3. Quick `0.2s` color/background hover transitions on every interactive element.
4. Gentle fade-up reveals; blinking input cursor for "live" feel.
5. Easing `cubic-bezier(0.25,0.75,0.25,1)` for the polished, eager-but-smooth feel.

---

## 9. Components

- **KPI / stat card:** dark `#121212`/`#171717` fill, `12–16px` radius, big bold number + tracked-out uppercase label + colored delta (green/red).
- **Chart panel:** `32px` radius container, animated bars/lines, edge-faded with black gradient masks, indigo series color.
- **Primary button:** indigo `#6e68ee` fill, pill or `8–12px` radius, white medium label, `0.2s` hover.
- **Status pill:** small, semantic color (green/red/amber), `500px` radius.
- **Nav:** dark, Inter medium links (Home, Features, Advantages, Benefits, Pricing, Reviews), "Waitlist" CTA.
- **Data table:** tight rows, `14px` text, hairline `rgba(255,255,255,0.15)` dividers, edge-faded.
- **Input / search:** dark field, blinking cursor, subtle inset border.

---

## 10. Voice & Copy

Clear, benefit-led, analytical. Emphasizes clarity, insight, and decision-making.
- "The Future Of Finance Dashboards"
- "Scattered financial data into visual insights"
- "Powerful features for data driven decisions."
- "Everything you need to understand your data."
- "Simplified for Clarity"
- "Bring clarity to your financial data."

Sections: Hero (dashboard mockup) → problem (scattered data) → features → "everything you need" → simplified/clarity → results/testimonials → pricing → FAQ → waitlist CTA.

---

## 11. Recreation Checklist

- [ ] Near-black surfaces (`#030303` → `#171717`); white text with 85/60% opacity tiers; muted `#9c9c9c`.
- [ ] Inter only; **default weight 500**; base UI/data text **14px** at line-height **1**.
- [ ] Brand accent indigo `#6e68ee`; semantic green `#22cb58` / red `#ff5e5d` / amber `#ffbc4f` for data only.
- [ ] Tracked-out uppercase labels (`letter-spacing: 1.8px`).
- [ ] Radii hierarchy: `4px` data chips → `12–16px` cards → `32px` containers; `500px` pills.
- [ ] **Animated diagonal hatch lines** (`0.7s linear infinite`) + growing chart bars + blinking input cursor.
- [ ] `0.2s ease` color/bg hover transitions; special easing `cubic-bezier(0.25,0.75,0.25,1)`.
- [ ] Black linear/radial gradient masks to feather charts, tables, and marquees into the background.
- [ ] Subtle multi-layer shadows + hairline borders; indigo glow behind key charts.

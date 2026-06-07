# Altrix Design System — Reference Spec

Extracted from the Altrix Framer template HTML. Use this file instead of parsing the HTML.

---

## 1. Design Philosophy

**Dark-first. Minimal. Sharp. Vibrant accents on black.**

- Background is near-black `#0a0a0a` (not pure black). Everything lives on this.
- Surfaces / cards are also very dark — barely lifted from the background (no white cards, no glass).
- Typography is restrained: tight letter-spacing (−0.04em), 130% line height, weight 400–500 for body, 700 for headings.
- The only colour that "pops" is the accent: `rgb(0,153,255)` — electric blue. Used sparingly on links, CTAs, and interactive highlights.
- Borders are near-invisible hairlines: `#f5f5f5` at very low opacity (not white borders at full opacity).
- Corners are **sharp**: 4px, 10px, 11px. Never large rounded "pill cards". Pill shapes exist only for tags/badges.
- Animations are subtle: blur-in/fade text reveals (`filter: blur → 0, opacity: 0 → 1, 0.2s ease-out`), no bouncy springs.
- Spacing is generous but consistent: gap `10px` (tight), `20px` (default), `30–50px` (section internal), `100px` (between major sections).

---

## 2. Colour Palette

### Base surfaces
| Role | Value | Usage |
|---|---|---|
| Page background | `#0a0a0a` | `<body>` and outermost container |
| Card / panel | `#0a0a0a` with hairline border | Most content containers |
| Slightly lighter surface | `#111` / `#141414` | Hover states, nested elements |
| Muted text | `#737373` | Labels, eyebrows, secondary info |
| Subdued text | `#525252` | Tertiary text |
| Faint text | `#404040` | Disabled / placeholder |

### Foreground (text)
| Role | Value |
|---|---|
| Primary text | `#f5f5f5` (not pure white) |
| Secondary text | `#fafafa` at ~80% (contextual) |
| Muted / label | `#737373` |

### Accents (use sparingly)
| Role | Value | Usage |
|---|---|---|
| Primary accent / links | `rgb(0,153,255)` = `#09f` | CTAs, links, active states, highlights |
| Neon green | `#00ff5e` | Positive data, success indicators |
| Amber / yellow | `#ffb300` | Warning states, highlighted numbers |
| Cyan | `#00f7ff` | Secondary accent, data viz |
| Yellow-green | `#eeFF00` / `#ef0` | Tertiary accent (data viz only) |
| Hot pink | `#ff008c` | Error, negative, destructive |

### Borders
| Role | Value |
|---|---|
| Default hairline | `#f5f5f5` at `opacity: 0.04` (inset box-shadow trick: `inset 0 0 0 1px rgb(0,0,0)` at 4% opacity) |
| Visible border | `#f5f5f5` (solid, used on cards with `--border-*` vars) |
| Accent border | `rgba(0,153,255,0.15)` / `rgba(255,179,0,0.15)` / `rgba(0,247,255,0.15)` |

**Implementation**: Altrix uses `--border-color: #f5f5f5` with `--border-*-width: 1px` on all 4 sides via Framer vars, which renders as a thin neutral line on the dark background.

---

## 3. Typography

**Font**: Satoshi (primary). Fallback: Inter, system-ui. Monospace: Fragment Mono.

| Element | Size | Weight | Line Height | Letter Spacing |
|---|---|---|---|---|
| Hero headline (H1) | 55px | 700 | 120% | −0.04em |
| Large heading | 45px | 700 | 120% | −0.04em |
| Section heading | 35px | 700 | 130% | −0.04em |
| Sub-heading | 26px | 500–700 | 130% | −0.02em |
| Body / card title | 20px | 500 | 130% | −0.02em |
| Small body | 16px | 400 | 130% | 0 |
| Label / eyebrow | 13px | 500 | 130% | 0 |
| Mono code | 12px | 400 | 130% | 0 |
| Tiny label | 8px | 500 | 130% | 0 |

**Technique**: Words animate in individually with `filter:blur(0px); opacity:1; transition: filter 0.2s ease-out, opacity 0.2s ease-out` — a word-by-word reveal on scroll/load.

---

## 4. Layout & Spacing

**Max width**: `1200px` centered, `padding: 0 20px` on mobile.

**Section padding** (vertical rhythm):
- Between major page sections: `100px 0`
- Section internal (hero, feature grid): `70px 20px` or `100px 20px`
- Card internal padding: `20px` or `30px`
- Compact items (nav, tags): `20px` horizontal padding

**Grid gaps**:
- Tight (tags, inline): `10px`
- Default (card grid): `20px`
- Generous (feature sections): `30–40px`
- Large (section rows): `50px`
- Section-to-section: `100px`

**Column layout**: Flexbox-first. Full-width sections with `place-content: center flex-start`. No sidebar. Two-column grids in features (`flex: 1 0 0` + `gap: 40–50px`).

---

## 5. Components

### Navigation (Navbar)
- Position: `fixed`, `z-index: 10`, full width.
- Height: `46px` container for logo/links area.
- Background: `#fafafa` (Altrix uses a **light** nav on the dark page — it's a floating nav that contrasts). _Note: for a trading dashboard, use the dark variant: `#0a0a0a` with bottom border hairline._
- Logo: SVG or wordmark, left-aligned.
- Nav links: `flex-flow: row; gap: 30px`. Text: `#0a0a0a` (on light nav). No underlines. Hover: `color: #09f` transition `0.4s cubic-bezier(.44...)`.
- CTA button: `padding: 20px` center-aligned flex, border-radius `4px`.

### Buttons
- **Primary CTA**: background `#0a0a0a`, text `#fafafa`, border `1px solid #f5f5f5`, padding `20px`, radius `4px`, `gap: 10px`. On hover: border becomes `#09f`.
- **Secondary / ghost**: no background, border `1px solid #f5f5f5`, same radius.
- No rounded-pill buttons — corners are `4px` always.
- Icon buttons use `gap: 10px` inside flex container.

### Cards / Panels
- Background: matches `#0a0a0a` (cards are flush with the page, differentiated by border only).
- Border: `1px solid #f5f5f5` (all 4 sides).
- Border-radius: `10px` or `11px`.
- Box-shadow: `rgba(0,0,0,0.17) 0px 0.6px 1.57px -1.5px, rgba(0,0,0,0.14) 0px 2.29px 5.95px -3px, rgba(0,0,0,0.02) 0px 10px 26px -4.5px` — very subtle depth.
- Internal padding: `20–30px`.
- The inner "opacity overlay" for inset border on dark bg: `box-shadow: inset 0 0 0 1px rgb(0,0,0); opacity: 0.04`.

### Feature Grid (3-column)
- `gap: 20px`, each item `flex-flow: column; gap: 16px`.
- Icon: color matches accent (`#09f`, `#00ff5e`, `#ffb300`, `#00f7ff` — rotated per feature).
- Title: 20px / 500.
- Body: 16px / 400, color `#737373`.

### Code Block / Terminal (Altrix signature motif)
- Dark card with monospace syntax-highlighted code.
- Background: near-black, border `1px solid #f5f5f5`, radius `10px`.
- Font: Fragment Mono, 12–13px.
- Syntax colours: keywords `#09f`, strings `#00ff5e`, comments `#737373`, plain `#f5f5f5`.
- Used as a visual design element (not just functional), appears in hero and feature sections.

### Tags / Badges
- `border-radius: 4px` (chip shape — NOT pill).
- Border `1px solid` with accent colour at 15% opacity (e.g. `rgba(0,153,255,0.15)`).
- Background: accent at 5–8% opacity.
- Text: accent colour, 12–13px, weight 500.

### Accent colour pills (status indicators)
- Same as badges but may use different accent colours: neon green for active/success, amber for warning, pink for error.

### Links
- Colour: `rgb(0,153,255)` = `#09f`.
- Underline: yes (`text-decoration: underline`).
- Transition: `color .4s cubic-bezier(.44, 0, .56, 1)`.

---

## 6. Animations & Motion

All animations are **subtle and fast**. Nothing draws excessive attention.

| Animation | Technique | Duration |
|---|---|---|
| Text word reveal (on load/scroll) | `filter: blur(0) → blur(4px)` + `opacity: 1 → 0` in reverse (blur-in), per word via inline style | `0.2s ease-out` per word |
| Fade up (card/section enter) | `opacity: 0→1` + `translateY: 8px→0` | `0.3–0.5s ease-out` |
| Hover on links | `color → #09f` | `0.4s cubic-bezier(.44,0,.56,1)` |
| Button hover | border-color changes | `0.15–0.2s ease` |
| Gradient scroll | `linear-gradient(180deg, rgba(0,0,0,0) ...)` fade overlay at section bottoms | CSS static |
| Chart/graph glow | `box-shadow` or SVG filter on data lines with accent colour | Static + hover |

**No parallax. No heavy scroll effects. No bounce/spring.** Clean linear or ease-out only.

---

## 7. Section Structure (Page Anatomy)

Top-to-bottom:

```
[Navbar] fixed, z-10
[Hero]  100px vertical padding, headline (55px bold, tight tracking), subtext, two CTA buttons, code block visual
[Logos / Social proof]  simple horizontal list, muted colours, gap 50px
[Feature highlight]  two-column, large heading left + card right (code terminal)
[Feature grid]  3-column, icon + title + body per card, 20px gap
[Pricing / CTA panel]  full-width dark card, centred headline, button row
[FAQ accordion]  left-aligned questions, subtle separator lines
[Footer CTA]  centred, large heading, two buttons
[Footer links]  muted links, 30px gap, legal row
```

---

## 8. Do's and Don'ts for Implementation

### DO
- Use `#0a0a0a` as the single page background — no gradient backgrounds.
- Use `#f5f5f5` (near-white) as the default text colour for body text.
- Use `#737373` for all labels, eyebrows, secondary text.
- Use `#09f` (rgb 0,153,255) as the ONLY default accent — keep it monochromatic.
- Use the other neon colours (`#00ff5e`, `#ffb300`, `#00f7ff`) ONLY for specific data states (profit, warning, info).
- Keep border-radius at `4px` for chips/tags, `10–11px` for cards, `4px` for buttons.
- Use generous padding: `20px` minimum inside any card.
- Use `gap: 10px` for inline items, `20px` for card grids, `40–100px` for section spacing.
- Animate text with a blur-fade word-by-word effect on page entry.
- Keep transitions fast: `0.15–0.2s ease-out` for hover, `0.2–0.4s` for enters.
- Use Fragment Mono for any code/terminal/data snippets — this is a signature motif.

### DON'T
- Don't use purple, violet, or indigo at all. The palette is black + blue-cyan only.
- Don't use white (`#ffffff`) for backgrounds or cards — use `#0a0a0a` or `#111`.
- Don't use large rounded corners (>12px) on cards or panels.
- Don't use heavy drop shadows or glow effects behind entire sections.
- Don't use glass/frosted blur effects (no `backdrop-filter` cards).
- Don't use gradient backgrounds on panels.
- Don't use animation delays > 0.5s or spring physics.
- Don't use more than 2 accent colours on a single screen.
- Don't make the nav sticky with a frosted/blurred background — keep it clean and either transparent or solid `#0a0a0a`.

---

## 9. Responsive Behaviour

- Max content width: `1200px`.
- Mobile breakpoint: `≤ 809px` — single column, padding `0 20px`.
- Tablet: `810–1199px` — may remain two-column with reduced gap.
- Section padding halves on mobile: `50px 20px` instead of `100px 20px`.
- Nav collapses to hamburger or simplified link row on mobile.
- Feature grid: 3-col → 1-col on mobile.

---

## 10. Quick Token Reference (CSS Custom Properties)

```css
--bg:           #0a0a0a;
--surface:      #0a0a0a;   /* cards are same bg, border differentiates */
--surface-raised: #111111; /* subtle lift for hover/nested */
--border:       #f5f5f5;   /* hairline, used at 1px solid */
--text-primary: #f5f5f5;
--text-secondary: #a3a3a3;
--text-muted:   #737373;
--text-faint:   #525252;

--accent:       #0099ff;   /* rgb(0,153,255) — primary blue */
--accent-green: #00ff5e;   /* profit / success */
--accent-amber: #ffb300;   /* warning */
--accent-cyan:  #00f7ff;   /* info */
--accent-pink:  #ff008c;   /* error / danger */

--radius-chip:  4px;
--radius-card:  10px;
--radius-large: 11px;

--font-sans: "Satoshi", "Inter", system-ui, sans-serif;
--font-mono: "Fragment Mono", "Fira Code", monospace;

--shadow-card: rgba(0,0,0,0.17) 0px 0.6px 1.57px -1.5px,
               rgba(0,0,0,0.14) 0px 2.29px 5.95px -3px,
               rgba(0,0,0,0.02) 0px 10px 26px -4.5px;

--transition-hover: 0.15s ease;
--transition-enter: 0.3s ease-out;
```

import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        page:    '#070707',
        surface: '#0f0f0f',
        accent:  '#f24100',
        win:     '#22c55e',
        loss:    '#ef4444',
        warn:    '#f59e0b',
        halt:    '#7f1d1d',
        primary: '#f2f2f2',
        secondary:'#969696',
      },
      fontFamily: {
        ui:   ['Inter', 'system-ui', 'sans-serif'],
        mono: ['Fragment Mono', 'IBM Plex Mono', 'monospace'],
      },
      borderRadius: {
        sm: '4px', md: '8px', lg: '12px',
      },
    },
  },
  plugins: [],
}
export default config

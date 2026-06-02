/** @type {import('next').NextConfig} */
const nextConfig = {
  async redirects() {
    // Browsers request /favicon.ico by default; we ship SVG under public/.
    return [{ source: "/favicon.ico", destination: "/favicon.svg", permanent: false }];
  },
};

export default nextConfig;
